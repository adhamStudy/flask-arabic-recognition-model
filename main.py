from flask import Flask, jsonify, request
from flask_cors import CORS  # Add this import
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model = tf.keras.models.load_model('ClassicCNN_v1_28_0.914.keras')
image_size = (32, 32)

def preprocess_image_for_cnn(image_stream, image_size=(32, 32)):
    img = np.array(image_stream)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    img = cv2.bitwise_not(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

labels = {
    0: 'ع',  # ain
    1: 'ا',  # alef
    2: 'ب',  # beh
    3: 'ض',  # dad
    4: 'د',  # dal
    5: 'ظ',  # dhad
    6: 'ف',  # feh
    7: 'غ',  # ghain
    8: 'ح',  # hah
    9: 'ه',  # heh
    10: 'ج',  # jeem
    11: 'ك',  # kaf
    12: 'خ',  # khah
    13: 'ل',  # lam
    14: 'م',  # meem
    15: 'ن',  # noon
    16: 'ق',  # qaf
    17: 'ر',  # reh
    18: 'ص',  # sad
    19: 'س',  # seen
    20: 'ش',  # sheen
    21: 'ط',  # tah
    22: 'ت',  # teh
    23: 'ذ',  # thal
    24: 'ث',  # theh
    25: 'و',  # waw
    26: 'ي',  # yeh
    27: 'ز'   # zain
}

@app.route('/predictImage', methods=['POST'])
def predictImage():
    try:
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided"
            }), 400

        file = request.files['image']
        image = Image.open(file.stream)
        processed_image = preprocess_image_for_cnn(image)
        pred = model.predict(processed_image)
        
        # Get the top 2 predictions
        top_2_indices = np.argsort(pred[0])[-2:][::-1]  # Indices of top 2 predictions
        top_2_labels = [labels[i] for i in top_2_indices]  # Corresponding labels
        top_2_probabilities = [float(pred[0][i]) for i in top_2_indices]  # Corresponding probabilities

        # Format the top 2 predictions as a single string separated by a comma
        predictions_string = f"{top_2_labels[0]} ({top_2_probabilities[0]:.4f}), {top_2_labels[1]} ({top_2_probabilities[1]:.4f})"

        return jsonify({
            "success": True,
            "prediction": predictions_string  # Ensure the key is "prediction"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)