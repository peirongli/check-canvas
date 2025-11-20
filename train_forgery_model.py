#!/usr/bin/env python3
"""
Simple model training script for image forgery detection
This demonstrates how to train a basic CNN model that can be used with the web app.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path
import random

class SimpleForgeryDataset(Dataset):
    """Simple dataset for forgery detection training"""
    
    def __init__(self, authentic_dir, forged_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        # Load authentic images (label 0)
        if os.path.exists(authentic_dir):
            for img_file in os.listdir(authentic_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({
                        'path': os.path.join(authentic_dir, img_file),
                        'label': 0,
                        'type': 'authentic'
                    })
        
        # Load forged images (label 1)
        if os.path.exists(forged_dir):
            for img_file in os.listdir(forged_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append({
                        'path': os.path.join(forged_dir, img_file),
                        'label': 1,
                        'type': 'forged'
                    })
        
        print(f"Loaded {len(self.samples)} images")
        print(f"Authentic: {sum(1 for s in self.samples if s['label'] == 0)}")
        print(f"Forged: {sum(1 for s in self.samples if s['label'] == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(sample['path']).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            return image, sample['label'], sample['path']
            
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
            # Return a dummy image
            return torch.zeros(3, 224, 224), sample['label'], sample['path']

class SimpleForgeryDetector(nn.Module):
    """Simple CNN for forgery detection"""
    
    def __init__(self, num_classes=2):
        super(SimpleForgeryDetector, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model():
    """Train the forgery detection model"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("Loading datasets...")
    
    # Use CASIA2 dataset structure
    train_dataset = SimpleForgeryDataset(
        authentic_dir="CASIA2/Au",
        forged_dir="CASIA2/Tp",  # You would need to create this directory with forged images
        transform=transform
    )
    
    # For demo purposes, we'll create a simple dataset with just authentic images
    # and simulate some as forged by random assignment
    if len(train_dataset) < 10:
        print("Not enough data for training. Creating synthetic dataset...")
        # This is just for demonstration - in practice you'd use real forged images
        
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    
    # Initialize model
    model = SimpleForgeryDetector().to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    num_epochs = 5  # Small number for demo
    print(f"Starting training for {num_epochs} epochs...")
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, paths) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1} completed - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        scheduler.step()
    
    print("Training completed!")
    
    # Save model
    model_path = "models/forgery_detector_demo.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'SimpleForgeryDetector',
        'num_classes': 2,
        'input_size': 224,
        'trained_epochs': num_epochs,
        'device': str(device)
    }, model_path)
    
    print(f"Model saved to: {model_path}")
    print("\nTo use this model with the web app:")
    print(f"1. Set MODEL_PATH={model_path} in your .env file")
    print("2. Restart the server")
    print("3. The web app will now use your trained model for detection!")

if __name__ == '__main__':
    train_model()