from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uuid
from datetime import datetime
import logging
import json
from google.cloud import storage, secretmanager
import firebase_admin
from firebase_admin import credentials, db, storage as firebase_storage
import os

app = Flask(__name__)

PROJECT_ID = "skripsi-f7dc5"
SECRET_NAME = "google-services"
BUCKET_NAME = "skripsi-f7dc5.firebasestorage.app"

# Fungsi untuk mengambil secret dari Google Secret Manager
def get_secret(secret_name, is_json=False):
    client = secretmanager.SecretManagerServiceClient()
    secret_path = f"projects/{PROJECT_ID}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": secret_path})

    if is_json:
        return json.loads(response.payload.data.decode("UTF-8"))
    else:
        return response.payload.data.decode("UTF-8")
   
# Inisialisasi Firebase
def initialize_firebase():
    try:
        service_account_info = get_secret(SECRET_NAME, is_json=True)
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred, {"databaseURL": get_secret("database"),
                                                "storageBucket": BUCKET_NAME})
        logging.info("Firebase initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Firebase: {repr(e)}")
        raise e

# Load model Google Cloud Storage
def load_model(local_model_path):
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob("models/model.keras")
        blob.download_to_filename(local_model_path)       
        return tf.keras.models.load_model(local_model_path)
    except Exception as e:
        logging.error(f"Error loading model from GCS: {repr(e)}")
        raise e

initialize_firebase()
model = load_model("/tmp/model.keras")

# Daftar kelas penyakit
class_names = ['Disease Free', 'Disease Free Fruit', 'Phytophthora', 'Red Rust', 'Scab', 'Styler End Rot']

# Fungsi untuk memproses gambar sebelum klasifikasi
def prepare_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Simpan gambar ke Firebase Storage
def save_image(image_file, user_id):
    filename = f"images/{user_id}/{uuid.uuid4()}.jpg"
    bucket = firebase_storage.bucket()
    blob = bucket.blob(filename)
    blob.upload_from_file(image_file, content_type='image/jpeg')
    return filename

# Ambil rekomendasi dari Firebase Realtime Database
def get_recommendation(disease):
    ref = db.reference('recommendations')
    recommendations = ref.get() or {}
    return recommendations.get(disease, "Tidak ada rekomendasi untuk penyakit ini.")

# Simpan hasil klasifikasi ke Firebase Database
def save_result(user_id, disease, confidence, image_path):
    ref = db.reference(f"history/{user_id}")
    new_entry = ref.push()
    new_entry.set({
        "diseaseName": disease,
        "accuracy": f"{confidence:.2%}",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "imagePath": image_path
    })

# Endpoint untuk klasifikasi gambar
@app.route('/classify', methods=['POST'])
def classify():
    if 'imageFile' not in request.files or 'user_id' not in request.form:
        return jsonify({'error': 'Gambar dan user_id harus disertakan'}), 400

    file = request.files['imageFile']
    user_id = request.form['user_id']

    try:
        image = Image.open(io.BytesIO(file.read()))
        processed_image = prepare_image(image)
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions)

        if confidence < 0.75:
            return jsonify({'error': 'Kepercayaan model terlalu rendah, coba gambar lain'}), 400

        disease = class_names[predicted_index]
        recommendation = get_recommendation(disease)

        file.seek(0)
        image_path = save_image(file, user_id)

        save_result(user_id, disease, confidence, image_path)

        return jsonify({
            'disease': disease,
            'confidence': f"{confidence:.2%}",
            'recommendations': recommendation,
            'imagePath': image_path
        })

    except Exception as e:
        logging.error(f"Terjadi kesalahan: {repr(e)}")
        return jsonify({'error': 'Terjadi kesalahan dalam klasifikasi'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
