#!/usr/bin/env python3
"""
Alternative Image Forgery Detection using Ensemble Methods
This approach combines multiple weak signals without relying on CLIP memorization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MultiModalDataset(Dataset):
    """Dataset that provides multiple types of features for ensemble learning"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def extract_frequency_features(self, image):
        """Extract frequency domain features"""
        import cv2
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift).astype(np.float32)
        
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Extract features from different frequency bands
        low_freq = magnitude[center_h-10:center_h+10, center_w-10:center_w+10]
        
        mid_freq_mask = np.zeros(magnitude.shape, dtype=np.float32)
        cv2.circle(mid_freq_mask, (center_w, center_h), 50, 1, -1)
        cv2.circle(mid_freq_mask, (center_w, center_h), 20, 0, -1)
        mid_freq = magnitude * mid_freq_mask
        
        high_freq_mask = np.ones(magnitude.shape, dtype=np.float32)
        cv2.circle(high_freq_mask, (center_w, center_h), 50, 0, -1)
        high_freq = magnitude * high_freq_mask
        
        freq_features = [
            np.mean(low_freq), np.std(low_freq), np.max(low_freq),
            np.mean(mid_freq), np.std(mid_freq), np.max(mid_freq),
            np.mean(high_freq), np.std(high_freq), np.max(high_freq)
        ]
        
        return freq_features
    
    def extract_texture_features(self, image):
        """Extract texture features using LBP"""
        import cv2
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Simple LBP implementation
        radius = 3
        n_points = 8
        lbp = np.zeros_like(gray)
        
        for i in range(radius, gray.shape[0] - radius):
            for j in range(radius, gray.shape[1] - radius):
                center = gray[i, j]
                binary = 0
                
                for p in range(n_points):
                    angle = 2 * np.pi * p / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    
                    if gray[x, y] >= center:
                        binary |= (1 << p)
                
                lbp[i, j] = binary
        
        # LBP histogram
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        lbp_hist = lbp_hist / np.sum(lbp_hist)
        
        return lbp_hist[:16].tolist()  # Top 16 bins
    
    def extract_edge_features(self, image):
        """Extract edge-based features"""
        import cv2
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Sobel edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_direction = np.arctan2(sobel_y, sobel_x)
        
        edge_features = [
            np.mean(edge_magnitude),
            np.std(edge_magnitude),
            np.max(edge_magnitude),
            np.mean(edge_direction),
            np.std(edge_direction)
        ]
        
        return edge_features
    
    def extract_color_features(self, image):
        """Extract color-based features"""
        import cv2
        
        if len(image.shape) == 3:
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            color_features = []
            
            # HSV statistics
            for i in range(3):
                channel = hsv[:, :, i]
                color_features.extend([np.mean(channel), np.std(channel)])
            
            # LAB statistics  
            for i in range(3):
                channel = lab[:, :, i]
                color_features.extend([np.mean(channel), np.std(channel)])
            
            # RGB statistics
            for i in range(3):
                channel = image[:, :, i]
                color_features.extend([np.mean(channel), np.std(channel)])
            
            return color_features
        else:
            # Grayscale
            return [np.mean(image), np.std(image)] * 9  # Pad to match expected size
    
    def extract_cnn_features(self, image, cnn_model):
        """Extract CNN features using a pre-trained model"""
        if self.transform:
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                features = cnn_model(image_tensor)
                
            return features.squeeze().numpy().tolist()
        else:
            return [0.0] * 512  # Default size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Extract multiple types of features
        freq_features = self.extract_frequency_features(image_np)
        texture_features = self.extract_texture_features(image_np)
        edge_features = self.extract_edge_features(image_np)
        color_features = self.extract_color_features(image_np)
        
        # Combine all features
        combined_features = freq_features + texture_features + edge_features + color_features
        
        return torch.FloatTensor(combined_features), label

class EnsembleClassifier(nn.Module):
    """Neural network ensemble classifier"""
    
    def __init__(self, input_features=48, hidden_dim=256, num_models=3):
        super().__init__()
        self.num_models = num_models
        
        # Create multiple sub-networks
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2)
            ) for _ in range(num_models)
        ])
        
        # Ensemble fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_models * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Concatenate predictions
        combined = torch.cat(predictions, dim=1)
        
        # Final fusion
        output = self.fusion(combined)
        
        return output, predictions

class StackingEnsemble:
    """Stacking ensemble using multiple base models and a meta-learner"""
    
    def __init__(self):
        self.base_models = []
        self.meta_model = None
        self.is_fitted = False
    
    def add_base_model(self, model, name):
        """Add a base model to the ensemble"""
        self.base_models.append((name, model))
    
    def train_base_models(self, X_train, y_train):
        """Train all base models"""
        base_predictions = []
        
        for name, model in self.base_models:
            print(f"Training {name}...")
            if hasattr(model, 'fit'):
                # Scikit-learn model
                model.fit(X_train, y_train)
                pred = model.predict_proba(X_train)[:, 1]  # Probability of positive class
            else:
                # PyTorch model
                model.train()
                # Training logic for PyTorch model
                pred = self._train_pytorch_model(model, X_train, y_train)
            
            base_predictions.append(pred)
            print(f"{name} training completed")
        
        return np.column_stack(base_predictions)
    
    def _train_pytorch_model(self, model, X_train, y_train):
        """Train a PyTorch model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create simple dataset
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.LongTensor(y_train).to(device)
        
        # Simple training loop
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(3):  # Reduced epochs for speed
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            
        return probs[:, 1].cpu().numpy()
    
    def fit(self, X_train, y_train):
        """Fit the stacking ensemble"""
        # Train base models and get their predictions
        base_predictions = self.train_base_models(X_train, y_train)
        
        # Train meta-model on base model predictions
        self.meta_model = LogisticRegression(random_state=42)
        self.meta_model.fit(base_predictions, y_train)
        
        self.is_fitted = True
    
    def predict_proba(self, X_test):
        """Predict probabilities using the ensemble"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get base model predictions
        base_predictions = []
        
        for name, model in self.base_models:
            if hasattr(model, 'predict_proba'):
                # Scikit-learn model
                pred = model.predict_proba(X_test)[:, 1]
            else:
                # PyTorch model
                model.eval()
                with torch.no_grad():
                    # Determine device from model parameters
                    device = next(model.parameters()).device
                    X_tensor = torch.FloatTensor(X_test).to(device)
                    outputs = model(X_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred = probs[:, 1].cpu().numpy()
            
            base_predictions.append(pred)
        
        # Stack predictions
        stacked_predictions = np.column_stack(base_predictions)
        
        # Meta-model prediction
        final_proba = self.meta_model.predict_proba(stacked_predictions)
        
        return final_proba
    
    def predict(self, X_test):
        """Predict class labels"""
        proba = self.predict_proba(X_test)
        return np.argmax(proba, axis=1)

def train_ensemble_model(image_paths, labels):
    """Train the ensemble model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split data - adjusted for smaller dataset
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )
    
    print(f"Training set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images") 
    print(f"Test set: {len(test_paths)} images")
    
    # Method 1: Neural Network Ensemble
    print("\n=== Training Neural Network Ensemble ===")
    
    # Create datasets
    train_dataset = MultiModalDataset(train_paths, train_labels)
    val_dataset = MultiModalDataset(val_paths, val_labels)
    test_dataset = MultiModalDataset(test_paths, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize ensemble model
    ensemble_model = EnsembleClassifier(input_features=48, hidden_dim=256, num_models=3).to(device)
    
    # Check if pre-trained model exists
    model_exists = os.path.exists('nn_ensemble_model.pth')
    
    if model_exists:
        print("Found existing nn_ensemble_model.pth, loading pre-trained model...")
        ensemble_model.load_state_dict(torch.load('nn_ensemble_model.pth', map_location=device))
        print("Pre-trained model loaded successfully.")
    else:
        print("No pre-trained model found, training from scratch...")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(ensemble_model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop - reduced epochs for faster development
        best_val_acc = 0
        for epoch in range(5):
            # Training
            ensemble_model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for features, labels in tqdm(train_loader, desc=f"NN Ensemble - Epoch {epoch+1}"):
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs, individual_outputs = ensemble_model(features)
                loss = criterion(outputs, labels)
                
                # Add diversity loss to encourage different models
                diversity_loss = 0
                for i in range(len(individual_outputs)):
                    for j in range(i+1, len(individual_outputs)):
                        # Cosine similarity between predictions
                        cos_sim = nn.functional.cosine_similarity(
                            individual_outputs[i], individual_outputs[j], dim=1
                        ).mean()
                        diversity_loss += cos_sim
                
                total_loss = loss - 0.1 * diversity_loss  # Encourage diversity
                total_loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation
            ensemble_model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    
                    outputs, _ = ensemble_model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            scheduler.step(val_loss / len(val_loader))
            
            print(f"NN Ensemble - Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(ensemble_model.state_dict(), 'nn_ensemble_model.pth')
                print(f"Saved best NN ensemble model with validation accuracy: {val_acc:.4f}")
    
    # Method 2: Stacking Ensemble with Traditional ML Models
    print("\n=== Training Stacking Ensemble ===")
    
    # Extract features for traditional ML models
    print("Extracting features for traditional models...")
    
    # Use the same feature extraction but convert to numpy
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    
    # Extract features for training set
    for i, (img_path, label) in enumerate(zip(train_paths, train_labels)):
        if i % 20 == 0:
            print(f"Processing training image {i+1}/{len(train_paths)}")
        
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Extract features
        freq_features = train_dataset.extract_frequency_features(image_np)
        texture_features = train_dataset.extract_texture_features(image_np)
        edge_features = train_dataset.extract_edge_features(image_np)
        color_features = train_dataset.extract_color_features(image_np)
        
        combined_features = freq_features + texture_features + edge_features + color_features
        
        X_train.append(combined_features)
        y_train.append(label)
    
    # Extract features for validation set
    for i, (img_path, label) in enumerate(zip(val_paths, val_labels)):
        if i % 10 == 0:
            print(f"Processing validation image {i+1}/{len(val_paths)}")
        
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Extract features
        freq_features = train_dataset.extract_frequency_features(image_np)
        texture_features = train_dataset.extract_texture_features(image_np)
        edge_features = train_dataset.extract_edge_features(image_np)
        color_features = train_dataset.extract_color_features(image_np)
        
        combined_features = freq_features + texture_features + edge_features + color_features
        
        X_val.append(combined_features)
        y_val.append(label)
    
    # Extract features for test set
    for i, (img_path, label) in enumerate(zip(test_paths, test_labels)):
        if i % 10 == 0:
            print(f"Processing test image {i+1}/{len(test_paths)}")
        
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Extract features
        freq_features = train_dataset.extract_frequency_features(image_np)
        texture_features = train_dataset.extract_texture_features(image_np)
        edge_features = train_dataset.extract_edge_features(image_np)
        color_features = train_dataset.extract_color_features(image_np)
        
        combined_features = freq_features + texture_features + edge_features + color_features
        
        X_test.append(combined_features)
        y_test.append(label)
    
    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Create stacking ensemble
    stacking_ensemble = StackingEnsemble()
    
    # Add base models
    stacking_ensemble.add_base_model(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")
    stacking_ensemble.add_base_model(SVC(probability=True, random_state=42), "SVM")
    stacking_ensemble.add_base_model(LogisticRegression(random_state=42, max_iter=1000), "Logistic Regression")
    
    # Add neural network as base model
    nn_model = nn.Sequential(
        nn.Linear(48, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 2)
    )
    stacking_ensemble.add_base_model(nn_model, "Neural Network")
    
    # Train stacking ensemble
    print("Training stacking ensemble...")
    stacking_ensemble.fit(X_train, y_train)
    
    # Evaluate stacking ensemble
    stacking_predictions = stacking_ensemble.predict(X_test)
    stacking_proba = stacking_ensemble.predict_proba(X_test)[:, 1]
    stacking_acc = accuracy_score(y_test, stacking_predictions)
    stacking_auc = roc_auc_score(y_test, stacking_proba)
    
    print(f"Stacking Ensemble Results: Accuracy={stacking_acc:.4f}, AUC={stacking_auc:.4f}")
    
    # Evaluate NN ensemble on test set
    ensemble_model.load_state_dict(torch.load('nn_ensemble_model.pth', map_location=device))
    ensemble_model.eval()
    
    nn_predictions = []
    nn_probs = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs, _ = ensemble_model(features)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            
            nn_predictions.extend(predicted.cpu().numpy())
            nn_probs.extend(probs[:, 1].cpu().numpy())
    
    nn_acc = accuracy_score(y_test, nn_predictions)
    nn_auc = roc_auc_score(y_test, nn_probs)
    
    print(f"NN Ensemble Results: Accuracy={nn_acc:.4f}, AUC={nn_auc:.4f}")
    
    # Final ensemble: Combine both approaches
    print("\n=== Combining Both Ensemble Approaches ===")
    
    # Weighted average of predictions
    final_probs = 0.6 * np.array(nn_probs) + 0.4 * stacking_proba
    final_predictions = (final_probs > 0.5).astype(int)
    final_acc = accuracy_score(y_test, final_predictions)
    final_auc = roc_auc_score(y_test, final_probs)
    
    print(f"Final Combined Ensemble Results: Accuracy={final_acc:.4f}, AUC={final_auc:.4f}")
    
    # Generate comprehensive report
    print("\n" + "="*60)
    print("ENSEMBLE METHOD COMPARISON REPORT")
    print("="*60)
    
    print(f"{'Method':<25} {'Accuracy':<12} {'AUC Score':<12}")
    print("-" * 50)
    print(f"{'NN Ensemble':<25} {nn_acc:<12.4f} {nn_auc:<12.4f}")
    print(f"{'Stacking Ensemble':<25} {stacking_acc:<12.4f} {stacking_auc:<12.4f}")
    print(f"{'Combined Ensemble':<25} {final_acc:<12.4f} {final_auc:<12.4f}")
    print("-" * 50)
    
    print(f"\nBest performing method: {'Combined Ensemble' if final_auc >= max(nn_auc, stacking_auc) else 'NN Ensemble' if nn_auc >= stacking_auc else 'Stacking Ensemble'}")
    
    return {
        'nn_ensemble': {'accuracy': nn_acc, 'auc': nn_auc, 'predictions': nn_predictions, 'probabilities': nn_probs},
        'stacking_ensemble': {'accuracy': stacking_acc, 'auc': stacking_auc, 'predictions': stacking_predictions, 'probabilities': stacking_proba},
        'combined_ensemble': {'accuracy': final_acc, 'auc': final_auc, 'predictions': final_predictions, 'probabilities': final_probs},
        'true_labels': y_test
    }

def load_casia2_data(data_dir="CASIA2", max_images=1000):
    """Load CASIA2 dataset paths and labels"""
    
    # Authentic images
    auth_dir = os.path.join(data_dir, "Au")
    auth_images = []
    if os.path.exists(auth_dir):
        auth_images = glob.glob(os.path.join(auth_dir, "*.jpg"))
    
    # Tampered images  
    tp_dir = os.path.join(data_dir, "Tp")
    tp_images = []
    if os.path.exists(tp_dir):
        tp_images = glob.glob(os.path.join(tp_dir, "*.jpg"))
    
    # Limit to max_images total, maintaining class balance
    max_per_class = max_images // 2
    
    # Sample from each class
    import random
    random.seed(42)  # For reproducibility
    
    if len(auth_images) > max_per_class:
        auth_images = random.sample(auth_images, max_per_class)
    
    if len(tp_images) > max_per_class:
        tp_images = random.sample(tp_images, max_per_class)
    
    # Combine and create labels
    all_images = auth_images + tp_images
    labels = [0] * len(auth_images) + [1] * len(tp_images)  # 0 = authentic, 1 = tampered
    
    return all_images, labels

def main():
    """Main function to run ensemble approach"""
    
    print("=== Ensemble Methods for Image Forgery Detection ===")
    print("Combining multiple weak signals without CLIP memorization")
    
    # Load data
    print("\nLoading CASIA2 dataset...")
    image_paths, labels = load_casia2_data()
    
    if len(image_paths) == 0:
        print("No images found. Please check CASIA2 dataset path.")
        return
    
    print(f"Found {len(image_paths)} images: {sum(1 for l in labels if l == 0)} authentic, {sum(1 for l in labels if l == 1)} tampered")
    
    # Train ensemble model
    results = train_ensemble_model(image_paths, labels)
    
    print("\nEnsemble training completed!")
    print("This approach combines multiple feature types and models to avoid overfitting and memorization issues")
    
    return results

if __name__ == "__main__":
    results = main()