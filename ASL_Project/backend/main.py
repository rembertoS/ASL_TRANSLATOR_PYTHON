from flask import Flask , request, jsonify
from flask_cors import CORS
import os
from model.predictor import ASLPredictor

app = Flask(__name__)
CORS(app)

# Initialize the predictor
predictor = ASLPredictor()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Create uploads directory if it doesn't exist
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save the uploaded file
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)

    try:
        # Get prediction from the model
        prediction = predictor.predict(filepath)
        return jsonify({'output': prediction})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)












