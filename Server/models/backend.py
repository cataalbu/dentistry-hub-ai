from bottle import route, request, run, response
import uuid
import tensorflow as tf
from PIL import Image
from skimage import transform
import io
import numpy as np

class TMJDPredictor:
    
    def __init__(self):
        self.model = tf.keras.saving.load_model("my_model.h5")

    def predict(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        np_image = np.array(image) / 1.0  
        np_image = transform.resize(np_image, (160, 160, 3)) 
        
        np_image = np.expand_dims(np_image, axis=0)
        
        prediction = self.model.predict(np_image)
        return prediction

tmjd_predictor = TMJDPredictor()

@route('/api/tmjd', method='POST')
def do_prediction():
    prediction_id = str(uuid.uuid4())

    upload = request.files.get('upload')
    if not upload:
        return "No image uploaded.", 400

    image_bytes = io.BytesIO(upload.file.read())
    image = Image.open(image_bytes)

    prediction = tmjd_predictor.predict(image)

    response.content_type = 'application/json'
    return {'id': prediction_id, 'prediction': prediction.flatten().tolist()}

if __name__ == '__main__':
    run(host='localhost', port=8080)
