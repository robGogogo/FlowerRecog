from flask import Flask, request, render_template
from keras.models import load_model
import tensorflow as tf
import numpy as np
import os
import tempfile

app = Flask(__name__)
model = load_model('Flower_Recog_Model.h5')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def classify_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(180,180))
    img_array = tf.keras.utils.img_to_array(img)
    img_expanded = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_expanded)
    result = tf.nn.softmax(prediction[0])
    label = flower_names[np.argmax(result)]
    confidence = np.max(result) * 100
    return label, confidence

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            file.save(temp.name)
            label, confidence = classify_image(temp.name)
        os.remove(temp.name)  # Clean up
        return render_template('index.html', prediction=label, confidence=confidence)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
