import React, { useState, useCallback } from 'react';
import { Upload, AlertCircle, CheckCircle, XCircle, Loader2, Image as ImageIcon } from 'lucide-react';
import { Toaster, toast } from 'sonner';
import axios from 'axios';

interface DetectionResult {
  isForgery: boolean;
  confidence: number;
  message: string;
  heatmapUrl?: string;
  originalImageUrl: string;
  isInconclusive?: boolean;
}

const ForgeryDetection: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>('');
  const [isDetecting, setIsDetecting] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [selectedModel, setSelectedModel] = useState<File | null>(null);
  const [modelName, setModelName] = useState<string>('ResNet50 (Default)');
  const [isUploadingModel, setIsUploadingModel] = useState(false);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedPretrainedModel, setSelectedPretrainedModel] = useState<string>('');
  const [isLoadingModels, setIsLoadingModels] = useState(false);

  const handleModelSelect = useCallback((file: File) => {
    if (!file.name.endsWith('.pt') && !file.name.endsWith('.pth')) {
      toast.error('Please select a valid PyTorch model file (.pt or .pth)');
      return;
    }

    if (file.size > 500 * 1024 * 1024) { // 500MB limit for model files
      toast.error('Model file size should be less than 500MB');
      return;
    }

    setSelectedModel(file);
    setModelName(file.name);
    toast.success('Model file selected successfully');
  }, []);

  const handleImageSelect = useCallback((file: File) => {
    if (!file.type.startsWith('image/')) {
      toast.error('Please select a valid image file');
      return;
    }

    if (file.size > 10 * 1024 * 1024) { // 10MB limit
      toast.error('Image size should be less than 10MB');
      return;
    }

    setSelectedImage(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setResult(null);
  }, []);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleImageSelect(files[0]);
    }
  }, [handleImageSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      handleImageSelect(files[0]);
    }
  }, [handleImageSelect]);

  const detectForgery = async () => {
    if (!selectedImage) {
      toast.error('Please select an image first');
      return;
    }

    setIsDetecting(true);
    const formData = new FormData();
    formData.append('image', selectedImage);
    
    // Add model selection - prioritize uploaded model, then pre-trained model
    if (selectedModel) {
      formData.append('model', selectedModel);
    } else if (selectedPretrainedModel) {
      formData.append('modelPath', selectedPretrainedModel);
    }

    try {
      const response = await axios.post('http://localhost:3001/api/detect', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResult(response.data);
      toast.success('Detection completed successfully');
    } catch (error) {
      console.error('Detection error:', error);
      toast.error('Failed to detect forgery. Please try again.');
    } finally {
      setIsDetecting(false);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setPreviewUrl('');
    setResult(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
  };

  const clearModel = () => {
    setSelectedModel(null);
    setSelectedPretrainedModel('');
    setModelName('ResNet50 (Default)');
  };

  const loadAvailableModels = async () => {
    setIsLoadingModels(true);
    try {
      const response = await axios.get('http://localhost:3001/api/models');
      if (response.data.success) {
        setAvailableModels(response.data.models.map((model: any) => model.name));
      }
    } catch (error) {
      console.error('Failed to load available models:', error);
      toast.error('Failed to load available models');
    } finally {
      setIsLoadingModels(false);
    }
  };

  React.useEffect(() => {
    loadAvailableModels();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8 px-4">
      <Toaster position="top-right" />
      
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Image Forgery Detection
          </h1>
          <p className="text-lg text-gray-600">
            Upload an image to detect if it has been tampered with using AI
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
          <div className="mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Select Model (Optional)
            </h2>
            
            <div className="mb-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium text-gray-700">
                  Current Model: {modelName}
                </span>
                {(selectedModel || selectedPretrainedModel) && (
                  <button
                    onClick={clearModel}
                    className="text-red-600 hover:text-red-700 text-sm font-medium"
                  >
                    Remove Model
                  </button>
                )}
              </div>
              
              {/* Pre-trained Models Selection */}
              {availableModels.length > 0 && (
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Pre-trained Models:
                  </label>
                  <select
                    value={selectedPretrainedModel}
                    onChange={(e) => {
                      const model = e.target.value;
                      setSelectedPretrainedModel(model);
                      setSelectedModel(null); // Clear uploaded model if selecting pre-trained
                      setModelName(model);
                      toast.success(`Selected model: ${model}`);
                    }}
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    disabled={isLoadingModels}
                  >
                    <option value="">Select a pre-trained model...</option>
                    {availableModels.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                  {isLoadingModels && (
                    <p className="text-sm text-gray-500 mt-1">Loading available models...</p>
                  )}
                </div>
              )}
              
              <div className="text-center mb-2">
                <span className="text-sm text-gray-500">or</span>
              </div>
              
              <label className="cursor-pointer">
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-gray-400 transition-colors">
                  <Upload className="mx-auto h-8 w-8 text-gray-400 mb-2" />
                  <span className="text-blue-600 hover:text-blue-500 font-medium">
                    Click to upload your trained model
                  </span>
                  <input
                    type="file"
                    accept=".pt,.pth"
                    onChange={(e) => {
                      const files = e.target.files;
                      if (files && files[0]) {
                        handleModelSelect(files[0]);
                        setSelectedPretrainedModel(''); // Clear pre-trained selection if uploading
                      }
                    }}
                    className="hidden"
                  />
                </div>
              </label>
              <p className="text-sm text-gray-500 mt-2">
                Supports PyTorch models (.pt, .pth) - Max 500MB
              </p>
            </div>
          </div>

          <div className="mb-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Upload Image
            </h2>
            
            {!selectedImage ? (
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  dragActive
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-300 hover:border-gray-400'
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                <p className="text-gray-600 mb-2">
                  Drag and drop your image here, or
                </p>
                <label className="cursor-pointer">
                  <span className="text-blue-600 hover:text-blue-500 font-medium">
                    click to browse
                  </span>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileInput}
                    className="hidden"
                  />
                </label>
                <p className="text-sm text-gray-500 mt-2">
                  Supports JPG, PNG, GIF (Max 10MB)
                </p>
              </div>
            ) : (
              <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg">
                <div className="relative flex-shrink-0">
                  <img
                    src={previewUrl}
                    alt="Selected"
                    className="w-24 h-24 object-cover rounded-lg shadow-sm"
                  />
                  <button
                    onClick={clearImage}
                    className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 transition-colors shadow-md"
                  >
                    <XCircle className="h-4 w-4" />
                  </button>
                </div>
                <div className="flex-1">
                  <p className="text-sm text-gray-600 mb-1">
                    Selected image: {selectedImage.name}
                  </p>
                  <p className="text-xs text-gray-500">
                    Size: {(selectedImage.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
            )}
          </div>

          {selectedImage && (
            <div className="text-center">
              <button
                onClick={detectForgery}
                disabled={isDetecting}
                className="bg-blue-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors inline-flex items-center gap-2"
              >
                {isDetecting ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Detecting...
                  </>
                ) : (
                  <>
                    <ImageIcon className="h-5 w-5" />
                    Detect Forgery
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {result && (
          <div className="bg-white rounded-xl shadow-lg p-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Detection Result
            </h2>
            
            <div className="mt-6 p-6 rounded-lg bg-gray-50">
              <div className="flex items-center gap-3 mb-4">
                {result.isInconclusive ? (
                  <AlertCircle className="h-8 w-8 text-yellow-500" />
                ) : result.isForgery ? (
                  <AlertCircle className="h-8 w-8 text-red-500" />
                ) : (
                  <CheckCircle className="h-8 w-8 text-green-500" />
                )}
                <h3 className={`text-2xl font-bold ${
                  result.isInconclusive ? 'text-yellow-700' : 
                  result.isForgery ? 'text-red-700' : 'text-green-700'
                }`}>
                  {result.isInconclusive ? 'Analysis Inconclusive' :
                   result.isForgery ? 'Forgery Detected' : 'Authentic Image'}
                </h3>
              </div>
              
              <p className="text-gray-700 text-lg mb-4 leading-relaxed">{result.message}</p>
              
              <div className="mb-4">
                <div className="flex justify-between text-sm font-medium text-gray-600 mb-2">
                  <span>Confidence Level</span>
                  <span>{(result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full transition-all duration-700 ${
                      result.isInconclusive ? 'bg-yellow-500' :
                      result.isForgery ? 'bg-red-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ForgeryDetection;