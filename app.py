import os
from flask import Flask, request, render_template, send_file, url_for
import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from PIL import Image
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from datetime import datetime
import base64

app = Flask(__name__)
app.config['REPORT_FOLDER'] = "reports"
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

# ====== Load EfficientNetB0 model ======
MODEL_PATH = "models/EfficientNetB0_SavedModel"
model = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")

# ====== Preprocess image like training ======
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0  # normalize like training
    image = np.expand_dims(image, axis=0)
    return image

# ====== Predict function ======
def predict_image(processed_image):
    preds = model(processed_image)
    if isinstance(preds, dict):
        output_key = list(preds.keys())[0]
        value = preds[output_key].numpy()
    else:
        value = preds.numpy()

    # Sigmoid (binary)
    sigmoid_diag, sigmoid_conf = None, None
    if value.ndim == 2 and value.shape[1] == 1:
        sigmoid_conf = float(value[0][0]) * 100
        sigmoid_diag = "Pneumonia" if value[0][0] > 0.5 else "Normal"

    # Softmax (2-class)
    softmax_diag, softmax_conf = None, None
    if value.ndim == 2 and value.shape[1] == 2:
        class_index = int(np.argmax(value[0]))
        softmax_conf = float(np.max(value[0])) * 100
        softmax_diag = "Pneumonia" if class_index == 1 else "Normal"

    return {
        "sigmoid": {"diagnosis": sigmoid_diag, "confidence": sigmoid_conf},
        "softmax": {"diagnosis": softmax_diag, "confidence": softmax_conf}
    }

# ====== Flask routes ======
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file)
            processed_image = preprocess_image(image)
            preds = predict_image(processed_image)

            # Convert image to base64 for preview
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()

            # Generate PDF report (using sigmoid prediction)
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            report_name = f"Report_{timestamp}.pdf"
            report_path = os.path.join(app.config['REPORT_FOLDER'], report_name)
            generate_report(preds['sigmoid']['diagnosis'], preds['sigmoid']['confidence'], image, report_path)
            pdf_link = url_for('download_report', filename=report_name)

            return render_template('upload.html',
                                   preds=preds,
                                   pdf_link=pdf_link,
                                   img_data=img_str)

    return render_template('upload.html')

@app.route('/download/<filename>')
def download_report(filename):
    return send_file(os.path.join(app.config['REPORT_FOLDER'], filename), as_attachment=True)

# ====== PDF report ======
def generate_report(diagnosis, confidence, image, report_path):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, 750, "Pneumonia Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Report Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.line(100, 720, 500, 720)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 700, "Diagnosis Information")
    c.setFont("Helvetica", 12)
    c.drawString(100, 680, f"Diagnosis: {diagnosis}")
    c.drawString(100, 660, f"Confidence: {confidence:.2f}%")

    # Draw image
    temp_img_path = os.path.join(app.config['REPORT_FOLDER'], "temp_xray.png")
    image.save(temp_img_path)
    c.drawImage(temp_img_path, 100, 300, width=350, height=280)
    os.remove(temp_img_path)

    c.setFont("Helvetica", 10)
    c.drawString(100, 270, "⚠️ This is an AI-generated report. Please verify with a medical professional.")
    c.line(100, 260, 480, 260)

    c.showPage()
    c.save()

    with open(report_path, "wb") as f:
        f.write(buffer.getvalue())

# ====== Run Flask ======
if __name__ == '__main__':
    app.run(debug=True)
