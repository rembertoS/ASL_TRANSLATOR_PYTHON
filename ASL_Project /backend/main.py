import flask 
import Flask , requests, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

def upload_file():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}) , 400
    

    filepath = os.path.join("uploads" , file.filename)
    os.makedirs("uploads" , exist_ok= True)
    file.save(filepath)

    prediction = "Test Prediction"

    return jsonify({'output': prediction})

if __name__ == '__main__':
    app.run(debug=True)












