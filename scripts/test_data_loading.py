import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
import torch
from src.data.dataset import DeepfakeDataset
from src.data.transforms import get_train_transforms

def create_dummy_video(path):
    # Create a 1-second video at 30 fps
    height, width = 224, 224
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
    
    for _ in range(30):
        # Create a random frame
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

def test_dataset():
    # Setup dummy directories
    root = "dummy_data"
    os.makedirs(os.path.join(root, "train", "Real"), exist_ok=True)
    os.makedirs(os.path.join(root, "train", "Fake"), exist_ok=True)
    
    # Create dummy videos
    create_dummy_video(os.path.join(root, "train", "Real", "real_vid.mp4"))
    create_dummy_video(os.path.join(root, "train", "Fake", "fake_vid.mp4"))
    
    # Init dataset
    transform = get_train_transforms()
    dataset = DeepfakeDataset(root_dir=os.path.join(root, "train"), transform=transform, num_frames=5)
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test loading
    item, label = dataset[0]
    print(f"Item shape: {item.shape}") # Should be (5, 3, 224, 224)
    print(f"Label: {label}")
    
    # Clean up
    import shutil
    shutil.rmtree(root)
    print("Test passed and cleanup done.")

if __name__ == "__main__":
    test_dataset()
