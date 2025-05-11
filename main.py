from flask import Flask, jsonify, request
from flask_cors import CORS  # Add this import
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
model = None 
# Load the model
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
    0: 'عين',  # ain
    1: 'ألف',  # alef
    2: 'باء',  # beh
    3: 'ضاد',  # dad
    4: 'دال',  # dal
    5: 'ظاء',  # dhad
    6: 'فاء',  # feh
    7: 'غين',  # ghain
    8: 'حاء',  # hah
    9: 'هاء',  # heh
    10: 'جيم',  # jeem
    11: 'كاف',  # kaf
    12: 'خاء',  # khah
    13: 'لام',  # lam
    14: 'ميم',  # meem
    15: 'نون',  # noon
    16: 'قاف',  # qaf
    17: 'راء',  # reh
    18: 'صاد',  # sad
    19: 'سين',  # seen
    20: 'شين',  # sheen
    21: 'طاء',  # tah
    22: 'تاء',  # teh
    23: 'ذال',  # thal
    24: 'ثاء',  # theh
    25: 'واو',  # waw
    26: 'ياء',  # yeh
    27: 'زاي'   # zain
}

@app.route('/predictImage', methods=['POST'])
def predictImage():
       global model
    if model is None:
        model = tf.keras.models.load_model('ClassicCNN_v1_28_0.914.keras')
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

        # Format the top 2 predictions as a single string separated by a comma
        predictions_string = ", ".join(top_2_labels)

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
