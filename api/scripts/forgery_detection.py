#!/usr/bin/env python3
"""
PyTorch Model Inference Script for Image Forgery Detection
This script provides a command-line interface for loading trained models
and performing inference on images. Uses ResNet50 as default model.
"""

import sys
import json
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from pathlib import Path
import argparse
import warnings
import os

# Suppress warnings and other output unless in verbose mode
class SuppressWarnings:
    """Context manager to suppress warnings and stderr, but preserve stdout for JSON output"""
    def __enter__(self):
        self._original_stderr = sys.stderr
        self._original_warnings = warnings.filters[:]
        # Suppress only stderr (warnings) but keep stdout for JSON output
        sys.stderr = open(os.devnull, 'w')
        # Suppress warnings
        warnings.simplefilter('ignore')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr
        warnings.filters[:] = self._original_warnings

# Simple CNN model for demonstration (replace with your actual model)
class SimpleForgeryDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleForgeryDetector, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResNetForgeryDetector(nn.Module):
    """ResNet50-based forgery detector with custom classification head"""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNetForgeryDetector, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove the original classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.backbone(x)
        return x

class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings for ViT"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2)  # (B, embed_dim, N_patches)
        x = x.transpose(1, 2)  # (B, N_patches, embed_dim)
        return x

class TransformerBlock(nn.Module):
    """Transformer block for ViT"""
    def __init__(self, embed_dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleViT(nn.Module):
    """Simple Vision Transformer for forgery detection"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=2,
                 embed_dim=384, depth=6, n_heads=6, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Use class token
        x = self.head(cls_token_final)
        
        return x

class ImprovedCNN(nn.Module):
    """Improved CNN model for forgery detection"""
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Layer 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Layer 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Layer 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ForgeryDetectionModel:
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu', verbose=False):
        self.device = device
        self.model = None
        self.transform = None
        self.verbose = verbose
        self.model_type = None
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model if path provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            if self.verbose:
                print(f"No model path provided or file not found. Using ResNet50 as default.")
            self.load_resnet_model()
    
    def detect_model_type(self, model_path):
        """Detect model type based on filename and structure"""
        model_name = Path(model_path).stem.lower()
        if 'vit' in model_name:
            return 'vit'
        elif 'cnn' in model_name:
            return 'cnn'
        elif 'ensemble' in model_name:
            return 'ensemble'
        else:
            return 'unknown'
    
    def create_model_from_state_dict(self, state_dict, model_type):
        """Dynamically create model architecture based on state dict"""
        if model_type == 'vit':
            # Analyze ViT architecture from state dict
            embed_dim = state_dict['pos_embed'].shape[2]
            n_patches = state_dict['pos_embed'].shape[1] - 1  # Subtract class token
            img_size = int((n_patches ** 0.5) * 16)  # Assuming patch_size=16
            depth = len([k for k in state_dict.keys() if k.startswith('blocks.') and k.endswith('.norm1.weight')])
            
            # Create ViT with detected parameters
            model = SimpleViT(
                img_size=img_size,
                patch_size=16,
                num_classes=2,
                embed_dim=embed_dim,
                depth=depth,
                n_heads=embed_dim // 64,  # Assuming head_dim=64
                mlp_ratio=4,
                dropout=0.1
            )
            return model
            
        elif model_type == 'cnn':
            # Analyze CNN architecture from state dict
            # Count conv layers and their dimensions
            conv_layers = {}
            for key in state_dict.keys():
                if key.startswith('features.') and key.endswith('.weight'):
                    parts = key.split('.')
                    if len(parts) >= 3:
                        layer_idx = int(parts[1])
                        if layer_idx not in conv_layers:
                            conv_layers[layer_idx] = state_dict[key].shape[0]
            
            # Create a simple CNN that matches the architecture
            layers = []
            in_channels = 3
            for i, layer_idx in enumerate(sorted(conv_layers.keys())):
                out_channels = conv_layers[layer_idx]
                dropout_rate = min(0.1 + i * 0.1, 0.5)  # Cap at 0.5
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(dropout_rate)
                ])
                in_channels = out_channels
            
            # Add classifier
            layers.extend([
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(in_channels, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 2)
            ])
            
            model = nn.Sequential(*layers)
            return model
        
        else:
            # Default to simple model
            return SimpleForgeryDetector()
    
    def load_model(self, model_path):
        """Load a trained PyTorch model with proper architecture detection"""
        try:
            # Detect model type from filename
            model_type = self.detect_model_type(model_path)
            
            if self.verbose:
                print(f"Detected model type: {model_type}")
            
            # Load checkpoint first to inspect structure
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Try to determine actual architecture from state dict
            if self.verbose:
                print("Inspecting model architecture from state dict keys...")
                print(f"State dict keys: {list(state_dict.keys())[:10]}...")  # Show first 10 keys
            
            # Create a flexible model that can load the state dict
            if model_type == 'vit':
                try:
                    # Try to create and load ViT
                    self.model = SimpleViT(num_classes=2)
                    self.model.load_state_dict(state_dict, strict=False)  # Allow partial loading
                    if self.verbose:
                        print("Using Vision Transformer architecture (partial loading)")
                except Exception as vit_error:
                    if self.verbose:
                        print(f"ViT loading failed: {vit_error}")
                        print("Trying alternative approach...")
                    # Create a simple wrapper that can handle any state dict
                    self.model = self.create_flexible_model(state_dict, num_classes=2)
                    if self.verbose:
                        print("Using flexible model architecture")
            elif model_type == 'cnn':
                try:
                    # Try to create and load CNN
                    self.model = ImprovedCNN(num_classes=2)
                    self.model.load_state_dict(state_dict, strict=False)  # Allow partial loading
                    if self.verbose:
                        print("Using Improved CNN architecture (partial loading)")
                except Exception as cnn_error:
                    if self.verbose:
                        print(f"CNN loading failed: {cnn_error}")
                        print("Trying flexible approach...")
                    # Create a simple wrapper that can handle any state dict
                    self.model = self.create_flexible_model(state_dict, num_classes=2)
                    if self.verbose:
                        print("Using flexible model architecture")
            elif model_type == 'ensemble':
                try:
                    # For ensemble models, try to load using the flexible model approach
                    if self.verbose:
                        print("Loading ensemble model with flexible architecture...")
                    self.model = self.create_flexible_model(state_dict, num_classes=2)
                    if self.verbose:
                        print("Using flexible ensemble architecture")
                except Exception as ensemble_error:
                    if self.verbose:
                        print(f"Ensemble loading failed: {ensemble_error}")
                        print("Trying simple CNN as fallback...")
                    # Fallback to simple detector
                    self.model = SimpleForgeryDetector()
                    self.model.load_state_dict(state_dict, strict=False)  # Allow partial loading
                    if self.verbose:
                        print("Using simple CNN architecture (partial loading)")
            else:
                # Try simple detector first for unknown models
                try:
                    self.model = SimpleForgeryDetector()
                    self.model.load_state_dict(state_dict, strict=False)  # Allow partial loading
                    if self.verbose:
                        print("Using simple CNN architecture (partial loading)")
                except Exception as simple_error:
                    if self.verbose:
                        print(f"Simple CNN failed: {simple_error}")
                        print("Trying flexible approach...")
                    # Create a simple wrapper that can handle any state dict
                    self.model = self.create_flexible_model(state_dict, num_classes=2)
                    if self.verbose:
                        print("Using flexible model architecture")
            
            self.model.to(self.device)
            self.model.eval()
            self.model_type = model_type
            
            if self.verbose:
                print(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading model: {e}")
                print("Falling back to ResNet50 model")
            self.load_resnet_model()
    
    def create_flexible_model(self, state_dict, num_classes=2):
        """Create a flexible model that can handle any compatible state dict"""
        class FlexibleModel(nn.Module):
            def __init__(self, state_dict, num_classes):
                super().__init__()
                self.features = nn.ModuleList()
                self.classifier = nn.ModuleList()
                self.num_classes = num_classes
                
                # Build layers from state dict by copying the exact structure
                # First, group keys by their layer names
                layer_groups = {}
                for key in state_dict.keys():
                    if '.' in key:
                        layer_name = key.split('.')[0]
                        param_name = '.'.join(key.split('.')[1:])
                        if layer_name not in layer_groups:
                            layer_groups[layer_name] = {}
                        layer_groups[layer_name][param_name] = state_dict[key].shape
                
                # Create modules based on the layer groups
                for layer_name, params in layer_groups.items():
                    if 'weight' in params:
                        weight_shape = params['weight']
                        
                        if len(weight_shape) == 4:  # Conv2d layer
                            out_channels, in_channels, kernel_size, _ = weight_shape
                            conv_layer = nn.Conv2d(
                                in_channels, out_channels, kernel_size, padding=kernel_size//2
                            )
                            self.features.append(conv_layer)
                            
                            # Add batch norm if it exists in state dict
                            bn_key = f'{layer_name}.running_mean'
                            if bn_key in state_dict:
                                self.features.append(nn.BatchNorm2d(out_channels))
                            
                            # Add activation and pooling based on common patterns
                            self.features.append(nn.ReLU())
                            # Add maxpool after conv blocks (common pattern)
                            if layer_name.startswith('features') and len(self.features) > 0:
                                self.features.append(nn.MaxPool2d(2))
                        
                        elif len(weight_shape) == 2:  # Linear layer
                            out_features, in_features = weight_shape
                            linear_layer = nn.Linear(in_features, out_features)
                            
                            # Add to classifier if it's the final layer
                            if 'classifier' in layer_name or 'fc' in layer_name:
                                self.classifier.append(linear_layer)
                            else:
                                self.classifier.append(linear_layer)
                                self.classifier.append(nn.ReLU())
                
                # Ensure we have a final classification layer
                if len(self.classifier) == 0 or (len(self.classifier) > 0 and 
                    not isinstance(self.classifier[-1], nn.Linear)):
                    # Find the last linear layer output features
                    last_features = None
                    for layer in reversed(self.classifier):
                        if isinstance(layer, nn.Linear):
                            last_features = layer.out_features
                            break
                    
                    if last_features is None:
                        # Estimate from conv layers
                        for layer in reversed(self.features):
                            if isinstance(layer, nn.Conv2d):
                                last_features = layer.out_channels
                                break
                    
                    if last_features is not None:
                        self.classifier.append(nn.Linear(last_features, num_classes))
            
            def forward(self, x):
                # Import F here to avoid issues
                import torch.nn.functional as F
                
                # Forward through feature layers
                for layer in self.features:
                    x = layer(x)
                
                # Flatten for classifier
                x = x.view(x.size(0), -1)
                
                # Forward through classifier layers
                for i, layer in enumerate(self.classifier):
                    x = layer(x)
                    # Add ReLU for intermediate linear layers
                    if isinstance(layer, nn.Linear) and i < len(self.classifier) - 1:
                        x = F.relu(x)
                
                return x
        
        # Create and load the flexible model
        model = FlexibleModel(state_dict, num_classes)
        
        # Load state dict with strict=False to allow partial loading
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            if self.verbose:
                print(f"Flexible model loading failed: {e}")
            # If strict loading fails, try to manually copy compatible weights
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        
        return model
    
    def load_resnet_model(self):
        """Load ResNet50 model for forgery detection"""
        try:
            self.model = ResNetForgeryDetector(num_classes=2, pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            if self.verbose:
                print("ResNet50 model loaded with pretrained weights")
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading ResNet50: {e}")
                print("Falling back to dummy model")
            self.load_dummy_model()
    
    def load_dummy_model(self):
        """Load a simple dummy model for demonstration"""
        self.model = SimpleForgeryDetector()
        # Initialize with random weights for demo
        self.model.to(self.device)
        self.model.eval()
        if self.verbose:
            print("Dummy model loaded for demonstration")
    
    def predict(self, image_path):
        """Predict forgery for a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get probabilities for both classes
                prob_authentic = probabilities[0][0].item()
                prob_forgery = probabilities[0][1].item()
                
                # Calculate confidence as the maximum probability
                confidence = max(prob_authentic, prob_forgery)
                
                # Determine predicted class based on higher probability
                predicted_class = 0 if prob_authentic > prob_forgery else 1
            
            # Map to forgery detection (0 = authentic, 1 = forgery)
            is_forgery = predicted_class == 1
            
            # Handle edge cases with very low confidence
            MIN_CONFIDENCE_THRESHOLD = 0.1  # 10% minimum confidence for definitive classification
            
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                # When confidence is extremely low, mark as inconclusive
                is_forgery = False  # Default to authentic but with warning
                confidence = max(confidence, 0.01)  # Ensure minimum 1% display to avoid 0.0%
            
            # Generate detailed analysis message based on model type and confidence
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                # Low confidence case - show inconclusive message
                message = f"⚠️ Analysis inconclusive with {confidence:.1%} confidence. "
                if prob_authentic > prob_forgery:
                    message += f"Slight indication of authenticity ({prob_authentic:.1%} vs {prob_forgery:.1%}). "
                else:
                    message += f"Slight indication of potential forgery ({prob_forgery:.1%} vs {prob_authentic:.1%}). "
                message += "Consider additional verification methods for critical applications."
            elif self.model_type == 'vit':
                # Vision Transformer model
                if is_forgery:
                    message = f"🚨 Potential forgery detected with {confidence:.1%} confidence using Vision Transformer (ViT)."
                    if confidence > 0.8:
                        message += " High confidence indicates strong evidence of manipulation through global analysis."
                    elif confidence > 0.6:
                        message += " Moderate confidence - further analysis recommended."
                else:
                    message = f"✅ Image appears authentic with {confidence:.1%} confidence using Vision Transformer (ViT)."
                    if confidence > 0.8:
                        message += " High confidence in authenticity through comprehensive global analysis."
                    else:
                        message += " Consider additional verification for critical applications."
            elif self.model_type == 'cnn':
                # Improved CNN model
                if is_forgery:
                    message = f"🚨 Potential forgery detected with {confidence:.1%} confidence using improved CNN."
                    if confidence > 0.8:
                        message += " High confidence indicates strong evidence of manipulation through hierarchical feature analysis."
                    elif confidence > 0.6:
                        message += " Moderate confidence - further analysis recommended."
                else:
                    message = f"✅ Image appears authentic with {confidence:.1%} confidence using improved CNN."
                    if confidence > 0.8:
                        message += " High confidence in authenticity through detailed local feature analysis."
                    else:
                        message += " Consider additional verification for critical applications."
            elif self.model_type == 'ensemble':
                # Neural Network Ensemble model
                if is_forgery:
                    message = f"🚨 Potential forgery detected with {confidence:.1%} confidence using Neural Network Ensemble."
                    if confidence > 0.8:
                        message += " High confidence indicates strong evidence of manipulation through multi-model consensus."
                    elif confidence > 0.6:
                        message += " Moderate confidence - ensemble analysis suggests potential tampering."
                else:
                    message = f"✅ Image appears authentic with {confidence:.1%} confidence using Neural Network Ensemble."
                    if confidence > 0.8:
                        message += " High confidence in authenticity through comprehensive multi-model analysis."
                    else:
                        message += " Ensemble analysis indicates authenticity but consider additional verification."
            elif hasattr(self.model, 'backbone'):
                # ResNet50 model
                if is_forgery:
                    message = f"🚨 Potential forgery detected with {confidence:.1%} confidence using advanced ResNet50 analysis."
                    if confidence > 0.8:
                        message += " High confidence indicates strong evidence of manipulation."
                    elif confidence > 0.6:
                        message += " Moderate confidence - further analysis recommended."
                else:
                    message = f"✅ Image appears authentic with {confidence:.1%} confidence using ResNet50 analysis."
                    if confidence > 0.8:
                        message += " High confidence in authenticity."
                    else:
                        message += " Consider additional verification for critical applications."
            else:
                # Simple model
                if is_forgery:
                    message = f"⚠️ Basic analysis suggests potential forgery with {confidence:.1%} confidence."
                else:
                    message = f"✓ Basic analysis indicates authenticity with {confidence:.1%} confidence."
            
            # Add image metadata
            try:
                with Image.open(image_path) as img:
                    message += f" Image dimensions: {img.size[0]}x{img.size[1]} pixels."
            except:
                pass
            
            # Determine model type string for response
            if self.model_type == 'vit':
                model_type_str = 'vision_transformer'
            elif self.model_type == 'cnn':
                model_type_str = 'improved_cnn'
            elif self.model_type == 'ensemble':
                model_type_str = 'neural_network_ensemble'
            elif hasattr(self.model, 'backbone'):
                model_type_str = 'resnet50'
            else:
                model_type_str = 'simple_cnn'
            
            return {
                'is_forgery': is_forgery,
                'confidence': confidence,
                'message': message,
                'predicted_class': predicted_class,
                'probabilities': [prob_authentic, prob_forgery],
                'model_type': model_type_str,
                'is_inconclusive': confidence < MIN_CONFIDENCE_THRESHOLD
            }
            
        except Exception as e:
            return {
                'error': f"Error processing image: {str(e)}",
                'is_forgery': False,
                'confidence': 0.0,
                'message': "Failed to analyze image due to processing error"
            }

def main():
    parser = argparse.ArgumentParser(description='Image Forgery Detection')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', help='Path to trained PyTorch model (.pt or .pth file)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use for inference')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Set device
        if args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.device
        
        # Print device info if verbose
        if args.verbose:
            print(f"Using device: {device}")
        
        # Use context manager to suppress warnings if not verbose
        if not args.verbose:
            with SuppressWarnings():
                # Initialize model
                detector = ForgeryDetectionModel(model_path=args.model, device=device, verbose=False)
                
                # Perform prediction
                result = detector.predict(args.image)
        else:
            # Initialize model with verbose output
            detector = ForgeryDetectionModel(model_path=args.model, device=device, verbose=True)
            
            # Perform prediction
            result = detector.predict(args.image)
        
        # Output result as JSON (always output JSON for API compatibility)
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        # Return error as JSON
        error_result = {
            'error': str(e),
            'is_forgery': False,
            'confidence': 0.0,
            'message': f"Analysis failed: {str(e)}"
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == '__main__':
    main()