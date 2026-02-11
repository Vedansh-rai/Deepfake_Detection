import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import DeepfakeDataset
from src.data.transforms import get_train_transforms, get_valid_transforms
from src.models.cnn_lstm import CNNLSTM

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data Loaders
    train_transform = get_train_transforms()
    valid_transform = get_valid_transforms()
    
    train_dataset = DeepfakeDataset(root_dir=os.path.join(args.data_dir, 'train'), transform=train_transform, num_frames=args.num_frames)
    valid_dataset = DeepfakeDataset(root_dir=os.path.join(args.data_dir, 'valid'), transform=valid_transform, num_frames=args.num_frames)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = CNNLSTM(num_classes=1).to(device)
    
    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for frames, labels in loop:
            frames = frames.to(device) # (B, T, C, H, W)
            labels = labels.float().unsqueeze(1).to(device) # (B, 1)
            
            optimizer.zero_grad()
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=correct/total)
            
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            loop = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]")
            for frames, labels in loop:
                frames = frames.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                
                outputs = model(frames)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                loop.set_postfix(val_loss=loss.item(), val_acc=val_correct/val_total)
        
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print("Saved best model.")
            
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/vedanshrai/Desktop/Deepfake Detection', help='Root directory of data')
    parser.add_argument('--checkpoint_dir', type=str, default='models', help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_frames', type=int, default=20, help='Number of frames per video')
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train(args)
