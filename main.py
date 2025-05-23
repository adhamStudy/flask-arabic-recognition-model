from flask import Flask, jsonify, request
from flask_cors import CORS  # Add this import
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import time


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
print("📦 Loading model...")
model = tf.keras.models.load_model('ClassicCNN_v1_28_0.914.keras')
print("✅ Model loaded.")
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
   try:
        print("📥 Received request")
        if 'image' not in request.files:
            print("❌ No image file provided")
            return jsonify({"success": False, "error": "No image file provided"}), 400

        file = request.files['image']
        image = Image.open(file.stream).convert('L')  # تحويل إلى grayscale
        image = image.resize(image_size)
        img_array = np.array(image)
        img_array = cv2.bitwise_not(img_array)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))  # إضافة بعدين

        print("✅ Image processed, predicting...")

        try:
            predictions = model.predict(img_array)
        except Exception as model_error:
            print(f"❌ Model prediction failed: {model_error}")
            return jsonify({"success": False, "error": "Model prediction failed"}), 500

        top_2_indices = np.argsort(predictions[0])[-2:][::-1]
        top_2_labels = [labels[i] for i in top_2_indices]
        predictions_string = ", ".join(top_2_labels)

        print(f"✅ Prediction complete: {predictions_string}")
        return jsonify({"success": True, "prediction": predictions_string})

    except Exception as e:
        print(f"❌ General error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500




@app.route('/predictImage1', methods=['POST'])
def predictImage1():
    return jsonify({"success": True, "prediction": "ميم, نون"})

@app.route('/predictImage2', methods=['POST'])
def predictImage2():
    try:
        print("📥 Received request in /predictImage2")
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400

        file = request.files['image']
        image = Image.open(file.stream).convert('L')  # تحويل للصورة الرمادية
        image = image.resize((32, 32))  # تغيير الحجم

        img_array = np.array(image)
        print(f"✅ Image shape: {img_array.shape}")

        # تحويل جزء من المصفوفة إلى قائمة (مثلاً أول 5 أسطر فقط لتجنب الضخامة)
        preview = img_array[:5].tolist()

        return jsonify({
            "success": True,
            "shape": img_array.shape,
            "preview": preview  # عرض جزء من المصفوفة
        })

    except Exception as e:
        print(f"❌ Error in /predictImage2: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Read port from Render
    app.run(host="0.0.0.0", port=port)
