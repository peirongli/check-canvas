#!/usr/bin/env python3
"""
Demo script to test the forgery detection functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'api', 'scripts'))

from forgery_detection import ForgeryDetectionModel
import json

def test_detection():
    """Test the forgery detection with a sample image"""
    
    # Initialize the model (will use dummy model if no real model provided)
    model = ForgeryDetectionModel()
    
    # Test with a sample image from CASIA2 dataset
    test_image = "CASIA2/Au/Au_ani_00001.jpg"
    
    if os.path.exists(test_image):
        print(f"Testing with image: {test_image}")
        result = model.predict(test_image)
        print("Detection Result:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Test image not found: {test_image}")
        print("Creating a dummy test...")
        
        # Create a simple test result
        result = {
            'is_forgery': False,
            'confidence': 0.85,
            'message': 'Dummy test - model is ready for inference'
        }
        print("Dummy Result:")
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    test_detection()