
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS

from torchvision import transforms, models
from PIL import Image
import io
import numpy as np
import threading
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os


import json

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas


# Cargar el archivo JSON con los nombres de los medicamentos
with open('medicamentos.json', 'r') as f:
    medicamentos_data = json.load(f)
    medicamentos = medicamentos_data['medicamentos']

# Definir y cargar el modelo PyTorch
model_path = 'resnet50.pth'
model = models.resnet50(pretrained=False)  # Definir la arquitectura del modelo

# Redefinir la capa fully connected para que coincida con el modelo entrenado
num_classes = 78  # Número de clases del modelo entrenado
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Cargar el estado del modelo
model.load_state_dict(torch.load(model_path))
model.eval()  # Modo de evaluación


# Definir la nueva normalización para imágenes en escala de grises
# Usar los valores calculados de media y desviación estándar
calculated_mean = [0.8722]
calculated_std = [0.2645]

# Definir transformaciones para las imágenes en escala de grises
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),  # Redimensionar a 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.8722,), (0.2645,))
])

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    
    # Aplicar las transformaciones restantes después de visualizar
    img_t = preprocess(img)
    img_np = np.array(img_t)  # Convertir a numpy array
    
    
    return img_t.unsqueeze(0)  # Añadir dimensión de lote

def predict_image(image_bytes):
    img_t = preprocess_image(image_bytes)
    with torch.no_grad():  # Desactivar el cálculo de gradientes
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
    return int(predicted.item())  # Convertir a entero estándar

# Ruta para predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    img_bytes = file.read()
    predicted_class_idx = predict_image(img_bytes)
    # Devolver el nombre del medicamento según el índice predicho
    if predicted_class_idx < len(medicamentos):
        predicted_class_idx= medicamentos[predicted_class_idx]
    else:
        predicted_class_idx= "Clase desconocida"
    
    # Devolver la clase predicha como JSON
    return jsonify({'pred': predicted_class_idx})

# Función para ejecutar Flask en un hilo separado
def run_flask():
    app.run(host='0.0.0.0', port=8080)


if __name__ =='__main__':
    app.run(debug=True,host="0.0.0.0",port=os.getenv("PORT",default=5000))