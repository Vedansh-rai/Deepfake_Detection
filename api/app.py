import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from src.models.cnn_lstm import CNNLSTM
from src.data.video_utils import extract_frames
from src.data.transforms import get_valid_transforms

app = Flask(__name__)

# specialized configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth')
NUM_FRAMES = 20

# Load Model
print(f"Loading model from {MODEL_PATH}...")
model = CNNLSTM(num_classes=1).to(DEVICE)

valid_transform = get_valid_transforms()

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Model loaded successfully.")
else:
    print("Warning: Model checkpoint not found. Using random weights.")

model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # Save temp file
        filename = file.filename
        filepath = os.path.join('/tmp', filename)
        file.save(filepath)
        
        try:
            # Inference Pipeline
            frames = extract_frames(filepath, num_frames=NUM_FRAMES)
            
            # Transform
            processed_frames = []
            for frame in frames:
                img = Image.fromarray(frame)
                img = valid_transform(img)
                processed_frames.append(img)
            
            # Stack and Batch
            frames_tensor = torch.stack(processed_frames).unsqueeze(0).to(DEVICE) # (1, T, C, H, W)
            
            with torch.no_grad():
                output = model(frames_tensor)
                prob = torch.sigmoid(output).item()
                
            prediction = "Fake" if prob > 0.5 else "Real"
            confidence = prob if prob > 0.5 else 1 - prob
            
            return jsonify({
                'prediction': prediction,
                'confidence': f"{confidence:.4f}",
                'probability': f"{prob:.4f}"
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
