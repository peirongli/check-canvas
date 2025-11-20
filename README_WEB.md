# Image Forgery Detection Web App

A modern web application for detecting image forgeries using AI and deep learning models. Built with React, TypeScript, Express, and PyTorch.

## Features

- 🖼️ **Image Upload**: Drag-and-drop or browse to upload images (JPG, PNG, GIF, max 10MB)
- 🤖 **Multi-Model AI Detection**: Advanced PyTorch models including ResNet50, Vision Transformer (ViT), CNN, and Neural Network Ensemble
- 📊 **Visual Results**: Shows detection results with confidence scores and detailed analysis
- 🔥 **Smart Model Selection**: Pre-trained models available with automatic fallback to ResNet50
- 📱 **Responsive Design**: Works on desktop and mobile devices
- ⚡ **Fast Processing**: Optimized inference with GPU acceleration support
- 🧠 **Advanced Architectures**: ResNet50, Vision Transformer, Improved CNN, and Ensemble models
- 🎯 **Custom Models**: Upload and use your own PyTorch models (.pt/.pth) with flexible architecture support
- 🔍 **Detailed Analysis**: Model-specific confidence messages and inconclusive detection handling
- 🛡️ **Robust Error Handling**: Graceful fallbacks and comprehensive error management

## Quick Start

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Install Python Dependencies** (for PyTorch model support)
   ```bash
   pip install -r api/scripts/requirements.txt
   ```

3. **Start the Application**
   ```bash
   npm run dev
   ```

4. **Open in Browser**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:3001

## Tech Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS, React Router, React Testing Library
- **Backend**: Express.js, Node.js, TypeScript, Concurrently for development
- **AI/ML**: PyTorch, Python integration via child processes, Multi-architecture support (CNN, ViT, Ensemble)
- **File Processing**: Multer for uploads, Sharp for image processing, Form-Data for API requests
- **State Management**: Zustand for React state, React hooks for component state
- **UI Components**: Lucide React icons, Sonner for notifications, CLSX for class management
- **Development Tools**: Vite for build tooling, ESLint for code quality, TypeScript compiler for type checking
- **Testing**: Jest testing framework with React Testing Library

## Usage

1. **Select Model** (Optional): Choose from pre-trained models or upload your own PyTorch model
2. **Upload an Image**: Drag and drop or click to browse for an image file (JPG, PNG, GIF, max 10MB)
3. **Detect Forgery**: Click the "Detect Forgery" button to analyze the image
4. **View Results**: See detailed detection results with confidence scores and model-specific analysis

**Enhanced Interface Features:**
- **Model Selection Dropdown**: Choose from available pre-trained models (ResNet50, ViT, CNN, Ensemble)
- **Custom Model Upload**: Upload your own PyTorch models with automatic architecture detection
- **Detailed Analysis**: Model-specific confidence messages and inconclusive detection handling
- **Visual Feedback**: Progress indicators and toast notifications for all operations

## Model Integration

The application now supports **multiple advanced AI architectures** for forgery detection, with intelligent model selection and flexible architecture support!

### Pre-trained Models Available

The system includes several pre-trained models in the `/models` directory:
- **ResNet50** (`cnn.pth`) - Advanced CNN with batch normalization and dropout
- **Vision Transformer** (`vit.pth`) - Transformer-based architecture for global image analysis  
- **Neural Network Ensemble** (`nn_ensemble.pth`) - Multi-model ensemble for robust detection

### Default ResNet50 Model

The system automatically uses ResNet50 as the fallback model, providing:
- **Immediate functionality**: Works out-of-the-box without additional setup
- **High accuracy**: ResNet50 architecture for robust feature extraction
- **Fast inference**: Optimized for quick predictions with GPU acceleration
- **Confidence scores**: Provides probability scores for authenticity/forgery

### Multi-Model Support Features

**Smart Model Selection:**
- Pre-trained models dropdown in the web interface
- Automatic model type detection from filename (vit, cnn, ensemble)
- Flexible architecture loading for custom models
- Graceful fallback to ResNet50 if model loading fails

**Advanced Model Architectures:**
- **Vision Transformer (ViT)**: Global attention mechanism for comprehensive image analysis
- **Improved CNN**: Multi-layer CNN with batch normalization and dropout
- **Neural Network Ensemble**: Consensus-based detection from multiple models
- **Flexible Model Loading**: Automatically adapts to your model's architecture

### Using Pre-trained Models

1. **Select from dropdown**: Choose from available pre-trained models in the web interface
2. **Automatic loading**: System loads the selected model automatically
3. **Model-specific analysis**: Get detailed results tailored to the model architecture

### Using Your Custom Model

1. **Upload your model**: Click "Upload your trained model" and select your `.pt` or `.pth` file
2. **Automatic detection**: System detects model type from filename and architecture
3. **Flexible loading**: Works with various PyTorch model architectures
4. **Architecture support**: Handles CNN, ViT, Ensemble, and custom architectures

### Enhanced Model Requirements

Your PyTorch model should:
- Accept 224x224 RGB images as input
- Output 2 classes (0: authentic, 1: forgery) 
- Return confidence scores via softmax
- Be saved as `.pt` or `.pth` file format
- **NEW**: Support for various architectures (CNN, ViT, Ensemble, Custom)

**Supported Model Types:**
```python
# Vision Transformer (ViT)
class SimpleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=2):
        # ViT implementation
        
# Improved CNN
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=2):
        # Multi-layer CNN with batch norm and dropout
        
# Neural Network Ensemble
class EnsembleModel(nn.Module):
    def __init__(self, num_classes=2):
        # Multi-model ensemble
        
# Custom Architecture (flexible loading)
class YourCustomModel(nn.Module):
    def __init__(self, num_classes=2):
        # Your custom architecture
```

## API Endpoints

### POST /api/detect
Detect forgery in an uploaded image with optional model selection.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: 
  - `image`: Image file (required)
  - `model`: Custom PyTorch model file (optional, .pt/.pth)
  - `modelPath`: Pre-trained model name (optional, e.g., "cnn.pth")

**Response:**
```json
{
  "isForgery": boolean,
  "confidence": number,
  "message": string,
  "originalImageUrl": string,
  "isInconclusive": boolean,
  "modelType": string,
  "probabilities": [number, number]
}
```

### GET /api/models
Get list of available pre-trained models.

**Response:**
```json
{
  "success": boolean,
  "models": [
    {
      "name": string,
      "path": string,
      "size": number|null
    }
  ]
}
```

### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "success": true,
  "message": "ok"
}
```

### GET /api/uploads/:filename
Serve uploaded images (for result display).

**Response:** Image file

## Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# PyTorch Model Configuration
MODEL_PATH=./models/your_model.pth
PYTHON_PATH=python3

# Server Configuration
PORT=3001
NODE_ENV=development
```

### Python Environment

The application uses Python for PyTorch model inference. Ensure you have:
- Python 3.7+
- PyTorch 1.9.0+
- Required Python packages (install with `pip install -r api/scripts/requirements.txt`)

## Development

### Project Structure

```
├── src/                    # Frontend React application
│   ├── components/         # React components (Empty, etc.)
│   ├── pages/             # Page components (ForgeryDetection, Home)
│   ├── hooks/             # Custom React hooks (useTheme)
│   ├── assets/            # Static assets (react.svg)
│   ├── lib/               # Utility functions (utils.ts)
│   └── __tests__/         # Test files (ForgeryDetection.test.tsx)
├── api/                    # Backend Express API
│   ├── routes/            # API routes (detection.ts, auth.ts)
│   ├── scripts/           # Python scripts for model inference
│   │   ├── forgery_detection.py  # Main PyTorch inference script
│   │   └── requirements.txt      # Python dependencies
│   ├── app.ts             # Express app configuration
│   ├── index.ts           # Server entry point
│   └── server.ts          # Server setup
├── uploads/               # Temporary image storage
├── models/                # PyTorch model files (cnn.pth, vit.pth, nn_ensemble.pth)
├── CASIA2/                # Sample image dataset (Au/ directory with test images)
├── .env                   # Environment configuration
├── .env.example           # Environment variables template
└── package.json           # Project dependencies and scripts
```

### Available Scripts

- `npm run dev` - Start development servers (frontend + backend concurrently)
- `npm run client:dev` - Start frontend development server only (Vite)
- `npm run server:dev` - Start backend development server only (Express with nodemon)
- `npm run build` - Build for production (TypeScript compilation + Vite build)
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint for code quality
- `npm run check` - Run TypeScript type checking
- **NEW**: Run tests with Jest and React Testing Library (configure test script as needed)

## Troubleshooting

### Common Issues

1. **Python not found**: Make sure Python is installed and accessible. Set `PYTHON_PATH` in `.env` if needed.

2. **PyTorch model loading fails**: 
   - Check that your model file exists and is accessible
   - Verify the model architecture matches the expected input/output format
   - Check console logs for detailed error messages
   - The system will automatically fall back to ResNet50 if your custom model fails
   - **NEW**: System now supports flexible architecture loading for various model types

3. **Image upload fails**:
   - Ensure the image is under 10MB
   - Check that the file format is supported (JPG, PNG, GIF)
   - Verify the uploads directory has write permissions

4. **Model selection issues**:
   - **Pre-trained models not showing**: Check that model files exist in `/models` directory
   - **Custom model upload fails**: Ensure file is `.pt` or `.pth` format and under 500MB
   - **Model loading errors**: System will automatically fall back to ResNet50
   - **Architecture detection**: System automatically detects CNN, ViT, Ensemble from filename

5. **Detection accuracy issues**:
   - Try different pre-trained models (ResNet50, ViT, CNN, Ensemble) for comparison
   - The default ResNet50 model provides good baseline performance
   - For domain-specific detection, consider training with your own dataset
   - Check model-specific confidence messages for detailed analysis
   - **NEW**: System handles inconclusive results with low confidence thresholds

6. **Multi-model specific issues**:
   - **ViT model slow**: Vision Transformer models may be slower on CPU, consider GPU acceleration
   - **Ensemble model large**: Neural Network Ensemble models may require more memory
   - **Custom architecture fails**: System will attempt flexible loading, fallback to simple CNN
   - **Model type detection**: Rename files with "vit", "cnn", or "ensemble" for better detection

7. **Development server issues**:
   - Ensure all dependencies are installed (`npm install` and `pip install -r api/scripts/requirements.txt`)
   - Check that ports 5173 (frontend) and 3001 (backend) are available
   - Verify TypeScript compilation with `npm run check`
   - **NEW**: Test individual components with `npm run client:dev` and `npm run server:dev`

### Model Training Tips

To train your own forgery detection model:

1. **Collect datasets**: Use authentic and forged image pairs with diverse scenarios
2. **Preprocess data**: Resize images to 224x224, normalize with ImageNet statistics
3. **Choose architecture**: 
   - **ResNet50**: Excellent for feature extraction and transfer learning
   - **Vision Transformer (ViT)**: Great for global image analysis and attention mechanisms
   - **Improved CNN**: Custom architectures with batch normalization and dropout
   - **Ensemble Models**: Combine multiple models for robust detection
4. **Train model**: Use appropriate loss functions (CrossEntropyLoss) and optimizers (AdamW)
5. **Evaluate**: Test on validation set and adjust hyperparameters
6. **Save model**: Export as .pt or .pth file with proper state dict format

**Architecture-Specific Tips:**
- **ViT Models**: Use appropriate patch sizes (16x16) and sufficient embedding dimensions
- **CNN Models**: Include batch normalization and dropout for better generalization
- **Ensemble Models**: Train diverse architectures and combine predictions
- **Custom Models**: Follow PyTorch conventions for state dict compatibility

### ResNet50 Default Model

The application includes a pre-configured ResNet50 model that:
- Uses ImageNet-pretrained weights for robust feature extraction
- Has a custom classification head for binary forgery detection
- Provides confidence scores for both authenticity and forgery classes
- Works immediately without any additional setup or training

This default model serves as an excellent baseline and can be used for:
- **Initial testing**: Verify the application works correctly
- **Proof of concept**: Demonstrate forgery detection capabilities
- **Development**: Build and test the application infrastructure
- **Comparison**: Benchmark against your custom trained models

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Inspired by VerifyVision-Pro project
- Built with modern web technologies
- Uses PyTorch for deep learning inference