from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the model
model_path = 'Skin Cancer.h5'
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = ['Benign', 'Malignant']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['file']
        # Read and preprocess the image
        img = Image.open(file.stream).resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        # Make predictions
        predictions = model.predict(img_array)
        # Get the predicted class label
        predicted_label = class_labels[np.argmax(predictions)]
        # Get the confidence score
        confidence = tf.reduce_max(tf.nn.softmax(predictions))
        return jsonify({'class': predicted_label, 'confidence': float(confidence)})

if __name__ == '__main__':
    app.run(debug=True)
