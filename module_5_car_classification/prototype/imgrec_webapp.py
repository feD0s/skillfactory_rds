import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

import io
from PIL import Image

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.models as M
import efficientnet.tfkeras as efn
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

from tempfile import NamedTemporaryFile

# Настройки
img_height = 250
img_width = 250

# Загрузка предобученной модели
print("Loading model")
model = keras.models.load_model('model_step4.hdf5')

# Задаем классы
class_names = ['Лада Приора','Ford Focus','ВАЗ 2114','ВАЗ 2110','ВАЗ 2107','Нива','Лада Калина','ВАЗ 2108','Volkswagen Passat','ВАЗ 21099']


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>')

def prediction(filename):
    
    img = keras.preprocessing.image.load_img(os.path.join('uploads', filename), target_size=(250, 250))

    # Преобразовываем в массив
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    # Предсказываем
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])    

    predictions = {
        "class1" : class_names[np.argmax(score)],
        "prob1" : 100 * np.max(score),
        }   

    return render_template('predict.html', predictions=predictions)  

app.run(host='0.0.0.0', port=5000)