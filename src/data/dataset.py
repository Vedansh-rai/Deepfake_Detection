import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from src.data.video_utils import extract_frames

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_frames=20, split='train'):
        """
        Args:
            root_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied on a sample.
            num_frames (int): Number of frames to extract from each video.
            split (str): 'train' or 'valid' or 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.num_frames = num_frames
        self.video_paths = []
        self.labels = []
        
        # Determine classes based on directory structure
        # Expected: root_dir/train/Fake, root_dir/train/Real
        # Or just root_dir/Fake, root_dir/Real depending on how the user passed root_dir
        
        # Let's handle standard ImageFolder-like structure
        # classes = ['Real', 'Fake'] usually
        self.classes = ['Real', 'Fake']
        self.class_to_idx = {'Real': 0, 'Fake': 1}
        
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            if not os.path.exists(cls_path):
                # Try lowercase if not found, or just skip
                if os.path.exists(os.path.join(root_dir, cls.lower())):
                     cls_path = os.path.join(root_dir, cls.lower())
                else:
                    continue
                    
            for video_name in os.listdir(cls_path):
                if video_name.startswith('.'): continue 
                video_path = os.path.join(cls_path, video_name)
                # Check if it's a file (video) or directory (some datasets have extracted frames)
                # Assuming video files for now based on user request "video frame extraction"
                if os.path.isfile(video_path):
                     self.video_paths.append(video_path)
                     self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        frames = extract_frames(video_path, self.num_frames)
        
        processed_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            if self.transform:
                img = self.transform(img)
            processed_frames.append(img)
            
        # Stack frames: (num_frames, C, H, W)
        frames_tensor = torch.stack(processed_frames)
        
        return frames_tensor, label
