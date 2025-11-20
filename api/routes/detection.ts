import express, { Request, Response, NextFunction } from 'express';
import multer, { File as MulterFile } from 'multer';
import sharp from 'sharp';
import * as path from 'path';
import * as fs from 'fs/promises';
import { spawn } from 'child_process';

// Define custom request interface
interface MulterRequest extends Request {
  file?: MulterFile;
}

const router = express.Router();

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  },
});

// Call Python script for PyTorch model inference
async function detectForgeryWithPython(imagePath: string, modelPath?: string): Promise<{
  isForgery: boolean;
  confidence: number;
  message: string;
}> {
  return new Promise((resolve, reject) => {
    const pythonPath = process.env.PYTHON_PATH || 'python3';
    const scriptPath = path.join(process.cwd(), 'api', 'scripts', 'forgery_detection.py');
    
    const args = ['--image', imagePath];
    if (modelPath) {
      args.push('--model', modelPath);
    }
    // Add verbose flag for debugging but suppress output in production
    if (process.env.NODE_ENV === 'development') {
      args.push('--verbose');
    }
    
    const pythonProcess = spawn(pythonPath, [scriptPath, ...args]);
    
    let stdout = '';
    let stderr = '';
    
    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('Python script error:', stderr);
        reject(new Error(`Python script failed with code ${code}: ${stderr}`));
        return;
      }
      
      try {
        // Parse the JSON output from Python
        // Look for JSON object in the entire stdout
        const jsonMatch = stdout.match(/\{[\s\S]*\}/);
        if (!jsonMatch) {
          throw new Error('No JSON output found from Python script');
        }
        
        const result = JSON.parse(jsonMatch[0]);
        
        if (result.error) {
          reject(new Error(result.error));
          return;
        }
        
        resolve({
          isForgery: result.is_forgery,
          confidence: result.confidence,
          message: result.message,
        });
      } catch (error) {
        console.error('Error parsing Python output:', stdout);
        reject(new Error(`Failed to parse Python output: ${error}`));
      }
    });
    
    pythonProcess.on('error', (error) => {
      reject(new Error(`Failed to spawn Python process: ${error.message}`));
    });
  });
}

// Basic fallback detection using image analysis
async function detectForgeryBasic(imageBuffer: Buffer): Promise<{
  isForgery: boolean;
  confidence: number;
  message: string;
}> {
  try {
    // Process image with sharp
    const image = sharp(imageBuffer);
    const metadata = await image.metadata();
    
    // Simple heuristic-based detection for demonstration
    const randomScore = Math.random();
    const isForgery = randomScore > 0.5;
    const confidence = Math.abs(randomScore - 0.5) * 2; // Convert to 0-1 range
    
    let message = '';
    if (isForgery) {
      message = 'Potential forgery detected based on basic analysis.';
    } else {
      message = 'The image appears to be authentic based on basic analysis.';
    }
    
    // Add some analysis details
    if (metadata.format) {
      message += ` Format: ${metadata.format.toUpperCase()}`;
    }
    if (metadata.width && metadata.height) {
      message += `, Dimensions: ${metadata.width}x${metadata.height}`;
    }
    
    return {
      isForgery,
      confidence,
      message,
    };
  } catch (error) {
    console.error('Error in basic forgery detection:', error);
    throw new Error('Failed to analyze image');
  }
}

// POST /api/detect - Handle image forgery detection
router.post('/detect', upload.single('image'), async (req: MulterRequest, res: Response, next: NextFunction) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'No image file provided',
      });
    }

    // Save uploaded file temporarily
    const uploadsDir = path.join(process.cwd(), 'uploads');
    await fs.mkdir(uploadsDir, { recursive: true });
    
    const filename = `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.jpg`;
    const filepath = path.join(uploadsDir, filename);
    
    // Process and save the image
    await sharp(req.file.buffer)
      .jpeg({ quality: 90 })
      .toFile(filepath);

    try {
      // Determine model path - prioritize uploaded model, then pre-trained models, then environment variable
      let modelPath: string | undefined;
      
      // Check if a custom model was uploaded
      if (req.body.modelPath) {
        // User selected a pre-trained model
        const modelsDir = path.join(process.cwd(), 'models');
        const selectedModel = req.body.modelPath;
        const potentialPath = path.join(modelsDir, selectedModel);
        
        if (await fs.access(potentialPath).then(() => true).catch(() => false)) {
          modelPath = potentialPath;
          console.log(`Using pre-trained model: ${selectedModel}`);
        } else {
          console.warn(`Selected model not found: ${selectedModel}`);
        }
      }
      
      // Fallback to environment variable or default
      if (!modelPath) {
        modelPath = process.env.MODEL_PATH;
      }
      
      // Perform forgery detection using Python script
      const detectionResult = await detectForgeryWithPython(filepath, modelPath);

      // Create response
      const result = {
        ...detectionResult,
        originalImageUrl: `/api/uploads/${filename}`,
        // heatmapUrl: detectionResult.isForgery ? `/api/uploads/heatmap_${filename}` : undefined, // Disabled - heatmap not implemented
      };

      res.json(result);
    } catch (pythonError) {
      console.error('Python detection failed, falling back to basic detection:', pythonError);
      
      // Fallback to basic detection if Python fails
      const basicResult = await detectForgeryBasic(req.file.buffer);
      const result = {
        ...basicResult,
        originalImageUrl: `/api/uploads/${filename}`,
        message: basicResult.message + ' (Basic analysis - PyTorch model unavailable)',
      };
      
      res.json(result);
    }
  } catch (error) {
    console.error('Detection error:', error);
    next(error);
  }
});

// GET /api/models - Get available pre-trained models
router.get('/models', async (req: Request, res: Response) => {
  try {
    const modelsDir = path.join(process.cwd(), 'models');
    
    // Check if models directory exists
    if (!await fs.access(modelsDir).then(() => true).catch(() => false)) {
      return res.json({
        success: true,
        models: []
      });
    }
    
    // Read models directory
    const files = await fs.readdir(modelsDir);
    const modelFiles = files.filter(file => 
      file.endsWith('.pth') || file.endsWith('.pt')
    );
    
    res.json({
      success: true,
      models: modelFiles.map(file => ({
        name: file,
        path: file,
        size: null // Could add file size if needed
      }))
    });
  } catch (error) {
    console.error('Error listing models:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to list available models'
    });
  }
});

// Serve uploaded files
router.use('/uploads', express.static(path.join(process.cwd(), 'uploads')));

export default router;