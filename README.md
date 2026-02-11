# Deepfake Detection System

This project implements a Deepfake Detection System using a CNN-LSTM architecture. It includes a training pipeline, a Flask API for serving predictions, and a simple web frontend.

## Dataset

The dataset used for training this model is available on Kaggle:
[Deepfake Dataset by Tushar Padhy](https://www.kaggle.com/datasets/tusharpadhy/deepfake-dataset)

### Dataset Setup
1. Download the dataset from the link above.
2. Extract the contents into the project root directory.
3. Ensure the directory structure is as follows:
   ```
   Deepfake Detection/
   ├── train/
   │   ├── Fake/
   │   └── Real/
   ├── valid/
   │   ├── Fake/
   │   └── Real/
   ├── test/ (optional)
   │   ├── Fake/
   │   └── Real/
   ```

## Project Structure
- `src/`: Source code for data loading, model architecture, and utilities.
- `scripts/`: Training scripts (`train_cnn_lstm.py`, `train_gan.py`).
- `api/`: Flask application and templates.
- `models/`: Directory where trained models are saved.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Vedansh-rai/Deepfake_Detection.git
    cd Deepfake_Detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model
To train the CNN-LSTM model:
```bash
python scripts/train_cnn_lstm.py --epochs 10 --batch_size 8
```

### Running the API
To start the Flask API and web interface:
```bash
python api/app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`.

## Author
Vedansh Rai
