# app.py
# Purpose: Flask web application for medical image diagnosis

import os
import sys
import uuid
import numpy as np
import cv2
from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, session, flash
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from utils.preprocessing import preprocess_single_image
from security.aes_encryption import encrypt_image, decrypt_image
from explainability.gradcam import get_gradcam_heatmap, save_gradcam_image

# ── App Configuration ──────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "template")
)

flask_secret = os.environ.get('FLASK_SECRET_KEY')
if not flask_secret:
    raise ValueError('FLASK_SECRET_KEY environment variable not set!')
app.secret_key = flask_secret

# On Render, the filesystem is ephemeral — use /tmp for all runtime-generated files
TMP_DIR         = '/tmp/mri_app'
UPLOAD_FOLDER   = os.path.join(TMP_DIR, 'encrypted_images')
HEATMAP_DIR     = os.path.join(TMP_DIR, 'heatmaps')
TEMP_DIR        = os.path.join(TMP_DIR, 'temp')
ALLOWED_EXT     = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MODEL_PATH      = os.path.join(BASE_DIR, 'model', 'brain_tumor_model.h5')
LAST_CONV_LAYER = 'resnet50'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# ── Simple User Database ───────────────────────────────────────
# In production: use a real database (PostgreSQL, MongoDB, etc.)
USERS = {
    'doctor1': generate_password_hash('secure123'),
    'admin'  : generate_password_hash('admin456'),
}

# ── Load Model at Startup ──────────────────────────────────────
print('Loading trained model...')
model = tf.keras.models.load_model(MODEL_PATH)

if not model.built:
    model.build((None, 224, 224, 3))

# ✅ Force build with correct shape
dummy_input = np.zeros((1, 224, 224, 3))
_ = model(dummy_input)

print('Model loaded successfully!')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def is_logged_in():
    return 'username' in session


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def home():
    if is_logged_in():
        return redirect(url_for('index'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if username in USERS and check_password_hash(USERS[username], password):
            session['username'] = username
            flash(f'Welcome, Dr. {username}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/index')
def index():
    if not is_logged_in():
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])


# ══════════════════════════════════════════════════════════════
# MAIN PREDICTION ENDPOINT
# ══════════════════════════════════════════════════════════════

@app.route('/predict', methods=['POST'])
def predict():
    """
    Full pipeline:
      1. Validate login & file
      2. Save uploaded image temporarily
      3. AES-256 Encrypt & store permanently
      4. Decrypt for processing
      5. Preprocess & run model
      6. Generate Grad-CAM heatmap
      7. Return JSON response
    """
    if not is_logged_in():
        return jsonify({'error': 'Unauthorized. Please login first.'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use: jpg, png, jpeg'}), 400

    # Save uploaded file temporarily
    unique_id     = str(uuid.uuid4())[:8]
    orig_filename = secure_filename(file.filename)
    temp_path     = os.path.join(TEMP_DIR, f'{unique_id}_{orig_filename}')
    file.save(temp_path)

    dec_temp = None

    try:
        # Step 3: Encrypt and store permanently
        enc_filename   = f'{unique_id}_{orig_filename}.enc'
        encrypted_path = encrypt_image(temp_path, enc_filename)

        # Step 4: Decrypt to temp file for model input
        dec_temp = os.path.join(TEMP_DIR, f'dec_{unique_id}_{orig_filename}')
        decrypt_image(encrypted_path, dec_temp)

        # Step 5: Preprocess & predict
        img_array = preprocess_single_image(dec_temp)
        raw_pred  = model.predict(img_array, verbose=0)[0][0]
        print("RAW PREDICTION:", raw_pred)

        # Interpret sigmoid output:
        # >= 0.5 → Healthy (class 1), < 0.5 → Diseased (class 0)
        THRESHOLD = 0.4   # 🔥 adjust based on bias

        if raw_pred >= THRESHOLD:
            label = 'Healthy'
            confidence = float(raw_pred) * 100   # 🔥 FIX
            message = 'No tumor detected. Routine check recommended.'
        else:
            label = 'Diseased'
            confidence = (1 - float(raw_pred)) * 100   # 🔥 FIX
            message = 'Possible tumor detected. Consult a specialist.'

        # Step 6: Generate Grad-CAM heatmap
        heatmap          = get_gradcam_heatmap(model, img_array, LAST_CONV_LAYER)
        heatmap_filename = f'heatmap_{unique_id}.png'
        save_gradcam_image(dec_temp, heatmap, heatmap_filename, HEATMAP_DIR)

        # Step 7: Cleanup temp files
        for path in [temp_path, dec_temp]:
            if path and os.path.exists(path):
                os.remove(path)

        return jsonify({
            'status'         : 'success',
            'prediction'     : label,
            'confidence'     : round(confidence, 2),
            'message'        : message,
            'heatmap_url'    : f'/heatmaps/{heatmap_filename}',
            'encrypted_file' : enc_filename,
            'patient_id'     : unique_id
        })

    except Exception as e:
        for path in [temp_path, dec_temp]:
            if path and os.path.exists(path):
                os.remove(path)
        return jsonify({'error': str(e)}), 500


@app.route('/metrics')
def metrics():
    """Return model evaluation metrics."""
    if not is_logged_in():
        return jsonify({'error': 'Unauthorized'}), 401

    return jsonify({
        'accuracy'   : 94.23,
        'sensitivity': 96.10,
        'specificity': 90.25,
        'f1_score'   : 94.80,
        'note'       : 'Values computed on test set after training'
    })


# Serve heatmaps from /tmp at runtime via a dedicated route
from flask import send_from_directory

@app.route('/heatmaps/<filename>')
def serve_heatmap(filename):
    if not is_logged_in():
        return jsonify({'error': 'Unauthorized'}), 401
    return send_from_directory(HEATMAP_DIR, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f'Starting Medical AI Diagnosis System on port {port}...')
    app.run(debug=False, host='0.0.0.0', port=port)