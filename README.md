# 🕵️‍♂️ Deepfake Detection System

## 📌 Project Overview
The **Deepfake Detection System** is a machine learning-based application designed to identify manipulated videos (deepfakes) with high accuracy. In an era where digital misinformation is rampant, this tool helps in verifying the authenticity of video content by analyzing spatial and temporal features.

This project utilizes a **CNN-LSTM (Convolutional Neural Network - Long Short-Term Memory)** architecture. The CNN extracts spatial features from individual video frames, while the LSTM captures temporal inconsistencies across frame sequences, making it robust against frame-by-frame manipulation.

## 🚀 Features
- **Deepfake Classification**: accurately classifies videos as "Real" or "Fake".
- **Spatiotemporal Analysis**: Combines CNN (ResNet50) and LSTM for robust detection.
- **Web Interface**: User-friendly Flask-based web app for uploading and testing videos.
- **REST API**: `/predict` endpoint for integrating detection capabilities into other applications.
- **Data Augmentation**: Scripts included for GAN-based data augmentation.

## 🛠️ Tech Stack
- **Language**: Python 3.8+
- **Deep Learning Framework**: PyTorch
- **Web Framework**: Flask
- **Data Processing**: OpenCV, NumPy, Pillow, Albumentations
- **Compute**: CUDA / MPS (Metal Performance Shaders for Mac) support

## 📂 Project Structure
```
Deepfake Detection/
├── api/
│   ├── app.py                # Flask API application
│   ├── templates/
│   │   └── index.html        # Web frontend
├── models/                   # Directory for saving trained models
├── scripts/
│   ├── train_cnn_lstm.py     # Main training script
│   ├── train_gan.py          # GAN training script (experimental)
│   └── test_data_loading.py  # Utility to verify data loading
├── src/
│   ├── data/
│   │   ├── dataset.py        # PyTorch Dataset class
│   │   ├── transforms.py     # Image transformations
│   │   └── video_utils.py    # Video frame extraction logic
│   ├── models/
│   │   └── cnn_lstm.py       # CNN-LSTM model architecture
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git ignore rules
```

## 📊 Dataset
The model is trained on the **Deepfake Dataset** by Tushar Padhy.
**[Download via Kaggle](https://www.kaggle.com/datasets/tusharpadhy/deepfake-dataset)**

### Data Organization
After downloading, extract the dataset into the project root. Your directory should look like this:
```
Deepfake Detection/
├── train/
│   ├── Fake/  # Contains fake videos/frames
│   └── Real/  # Contains real videos/frames
├── valid/
│   ├── Fake/
│   └── Real/
```

## ⚙️ Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Vedansh-rai/Deepfake_Detection.git
    cd Deepfake_Detection
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🏋️‍♂️ Training the Model

To train the CNN-LSTM model from scratch, run the following command:

```bash
python scripts/train_cnn_lstm.py --epochs 10 --batch_size 8 --learning_rate 0.0001
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 8)
- `--num_frames`: Number of frames to extract per video (default: 20)
- `--checkpoint_dir`: Directory to save model checkpoints (default: `models`)

_Note: The script automatically detects if a GPU (CUDA) or Mac MPS is available and uses it for faster training._

## 🌐 Running the Web Application

Once the model is trained (or if you have a pre-trained model in `models/best_model.pth`), you can launch the web interface:

1.  **Start the Flask Server**
    ```bash
    python api/app.py
    ```

2.  **Access the App**
    Open your web browser and go to: `http://127.0.0.1:5000`

3.  **Upload & Predict**
    - Click "Choose File" to select a video (`.mp4`, `.avi`, `.mov`).
    - Click "Analyze Video".
    - View the prediction ("Real" or "Fake") and the confidence score.

## 🧠 Model Architecture Details
The system uses a hybrid architecture:
1.  **ResNet50 (Pre-trained)**: Acts as the feature extractor. We remove the fully connected top layers and use the convolutional output to represent each frame.
2.  **LSTM (Long Short-Term Memory)**: Takes the sequence of feature vectors from ResNet50. It processes the temporal evolution of features to detect anomalies that occur over time (e.g., flickering, unnatural movements).
3.  **Classification Head**: A fully connected layer maps the final LSTM output to a binary probability score.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## 📜 License
This project is open-source and available under the MIT License.

## 👤 Author
**Vedansh Rai**
- [GitHub](https://github.com/Vedansh-rai)
