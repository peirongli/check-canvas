#!/usr/bin/env node

/**
 * Simple API test for the forgery detection endpoint
 */

import axios from 'axios';
import FormData from 'form-data';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function testDetectionAPI() {
  console.log('Testing forgery detection API...');
  
  try {
    // Create form data with a test image
    const formData = new FormData();
    
    // Try to use a real image from CASIA2 dataset
    const testImagePath = 'CASIA2/Au/Au_ani_00001.jpg';
    
    if (fs.existsSync(testImagePath)) {
      const imageBuffer = fs.readFileSync(testImagePath);
      formData.append('image', imageBuffer, {
        filename: 'test.jpg',
        contentType: 'image/jpeg'
      });
      
      console.log(`Using test image: ${testImagePath}`);
    } else {
      // Create a dummy image buffer for testing
      const dummyBuffer = Buffer.from('dummy image data');
      formData.append('image', dummyBuffer, {
        filename: 'test.jpg',
        contentType: 'image/jpeg'
      });
      
      console.log('Using dummy image data for testing');
    }
    
    // Make API request
    console.log('Sending request to http://localhost:3001/api/detect');
    
    const response = await axios.post('http://localhost:3001/api/detect', formData, {
      headers: {
        ...formData.getHeaders()
      },
      timeout: 30000 // 30 second timeout
    });
    
    console.log('Response received:');
    console.log(JSON.stringify(response.data, null, 2));
    
    // Validate response structure
    const result = response.data;
    if (typeof result.isForgery === 'boolean' && 
        typeof result.confidence === 'number' && 
        typeof result.message === 'string') {
      console.log('✅ API response structure is valid');
    } else {
      console.log('❌ API response structure is invalid');
    }
    
  } catch (error) {
    console.error('API test failed:', error.message);
    
    if (error.code === 'ECONNREFUSED') {
      console.log('Make sure the server is running: npm run dev');
    } else if (error.response) {
      console.log('Server responded with error:', error.response.status, error.response.data);
    }
  }
}

// Run the test
testDetectionAPI();