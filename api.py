from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import logging
import uuid
from datetime import datetime
from google.cloud import storage, secretmanager
import firebase_admin
from firebase_admin import credentials, db

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Flask app setup
app = Flask(__name__)

# Google Secret Manager client
secret_client = secretmanager.SecretManagerServiceClient()

# Retrieve secret from Secret Manager
def get_secret(secret_name, project_id):
    try:
        name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        response = secret_client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logging.error(f"Error accessing secret {secret_name}: {repr(e)}")
        raise e

# Initialize Firebase using secret from Secret Manager
def initialize_firebase():
    try:
        service_account_info = get_secret("firebase-credentials", "<your-project-id>")
        cred = credentials.Certificate(json.loads(service_account_info))
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://<your-database-name>.firebaseio.com/',
        })
    except Exception as e:
        logging.error(f"Error initializing Firebase: {repr(e)}")
        raise e

# Load model from GCS
def load_model(bucket_name, model_path):
    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(model_path)
        model_file = io.BytesIO()
        blob.download_to_file(model_file)
        model_file.seek(0)
        return tf.keras.models.load_model(model_file)
    except Exception as e:
        logging.error(f"Error loading model from GCS: {repr(e)}")
        return None

# Load recommendations from GCS
def load_recommend(bucket_name, recommendations_path):
    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(recommendations_path)
        recommendations_file = blob.download_as_text(encoding="utf-8")
        return json.loads(recommendations_file)
    except Exception as e:
        logging.error(f"Error loading recommendations from GCS: {repr(e)}")
        raise e

# Upload image to Google Cloud Storage
def save_image(image_file, filename, bucket_name):
    try:
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(filename)
        blob.upload_from_file(image_file, content_type='image/jpeg')
        blob.make_public()
        return blob.public_url
    except Exception as e:
        logging.error(f"Error uploading image to GCS: {repr(e)}")
        raise e

def save_result(user_id, date, disease, confidence, img_url):
    try:
        # Generate a unique key for the entry
        ref = db.reference('classification_history')
        new_entry = ref.push()
        
        # Save the classification result
        new_entry.set({
            'user_id': user_id,
            'date': date,
            'disease': disease,
            'confidence': confidence,
            'img_url': img_url
        })
        logging.info("Classification result saved to Firebase successfully.")
    except Exception as e:
        logging.error(f"Error saving to Firebase: {repr(e)}")
        raise e

# Prepare image for prediction
def prepare_image(image, target_size):
    try:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(target_size)
        image = np.array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logging.error(f"Error preparing image: {repr(e)}")
        raise e

# Initialize Google Cloud Storage and Firebase
project_id = "<your-project-id>"
gcs_client = storage.Client()
initialize_firebase()

# Load sensitive data from Secret Manager
bucket_name = get_secret("gcs-bucket-name", project_id)
model_path = "models/model.keras"
recommendations_path = "configs/recommendations.json"

# Load model and recommendations
model = load_model(bucket_name, model_path)
if not model:
    logging.error("Model failed to load. Please check the file in GCS.")

recommendations = load_recommend(bucket_name, recommendations_path)
if not recommendations:
    raise Exception("Recommendations failed to load. Please check the file in GCS.")

# Define disease classes
class_names = ['Disease Free', 'Disease Free Fruit', 'Phytophthora', 'Red Rust', 'Scab', 'Styler End Rot']

# Process classification request
@app.route('/classify', methods=['POST'])
def classify():
    if 'imageFile' not in request.files:
        logging.error("No image file provided in request.")
        return jsonify({'error': 'Image file not provided'}), 400

    file = request.files['imageFile']
    if file.filename == '':
        logging.error("Empty image file provided.")
        return jsonify({'error': 'Empty file'}), 400    

    try:
        user_id = request.form['user_id']
        # Read and process the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = prepare_image(image, target_size=(224, 224))
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        accuracy = np.max(predictions)

        # Define a confidence threshold
        threshold = 0.75
        if accuracy < threshold:
            raise Exception("Image classification confidence is too low. Please try another image.")

        predicted_class = class_names[predicted_class_index]
        recommendation = recommendations.get(predicted_class, "No recommendation available for this class.")

        # Save image to Google Cloud Storage
        filename = f"images/{uuid.uuid4()}.jpg"
        img_url = save_image(io.BytesIO(file.read()), filename, bucket_name)

        # Save classification results to Firebase
        save_result(user_id, datetime.now().isoformat(), predicted_class, f"{accuracy:.2%}", img_url)

        return jsonify({
            'disease': predicted_class,
            'confidence': f"{accuracy:.2%}",
            'recommendations': recommendation,
            'image_url': img_url
        })

    except Exception as e:
        logging.error(f"Prediction error: {repr(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
