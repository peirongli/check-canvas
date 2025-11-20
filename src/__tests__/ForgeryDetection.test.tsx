import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ForgeryDetection from '../pages/ForgeryDetection';

// Mock axios
jest.mock('axios', () => ({
  post: jest.fn(),
}));

// Mock URL.createObjectURL
global.URL.createObjectURL = jest.fn(() => 'mocked-url');
global.URL.revokeObjectURL = jest.fn();

describe('ForgeryDetection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders the component', () => {
    render(<ForgeryDetection />);
    
    expect(screen.getByText('Image Forgery Detection')).toBeInTheDocument();
    expect(screen.getByText('Upload an image to detect if it has been tampered with using AI')).toBeInTheDocument();
  });

  test('handles file upload', () => {
    render(<ForgeryDetection />);
    
    const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
    const input = screen.getByLabelText('click to browse');
    
    fireEvent.change(input, { target: { files: [file] } });
    
    expect(screen.getByAltText('Selected')).toBeInTheDocument();
  });

  test('shows error for non-image files', async () => {
    render(<ForgeryDetection />);
    
    const file = new File(['test'], 'test.txt', { type: 'text/plain' });
    const input = screen.getByLabelText('click to browse');
    
    fireEvent.change(input, { target: { files: [file] } });
    
    await waitFor(() => {
      expect(screen.getByText('Please select a valid image file')).toBeInTheDocument();
    });
  });

  test('shows detect button when image is selected', () => {
    render(<ForgeryDetection />);
    
    const file = new File(['test'], 'test.jpg', { type: 'image/jpeg' });
    const input = screen.getByLabelText('click to browse');
    
    fireEvent.change(input, { target: { files: [file] } });
    
    expect(screen.getByText('Detect Forgery')).toBeInTheDocument();
  });
});