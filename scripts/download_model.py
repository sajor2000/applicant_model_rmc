#!/usr/bin/env python3
"""Download pre-trained model files for the Rush AI Admissions System."""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib

# Model download configuration
MODEL_URL = "https://example.com/rush-ai-models/comprehensive_cascade_v1.pkl"  # Replace with actual URL
MODEL_NAME = "comprehensive_cascade_20250619.pkl"
MODEL_SIZE = 150 * 1024 * 1024  # 150 MB approximate
MODEL_HASH = "abc123def456"  # Replace with actual hash

MODELS_DIR = Path(__file__).parent.parent / "models"


def download_file(url: str, destination: Path, expected_size: int = None) -> bool:
    """Download a file with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save file
        expected_size: Expected file size in bytes
        
    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', expected_size or 0))
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return True
        
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


def verify_file(filepath: Path, expected_hash: str) -> bool:
    """Verify file integrity using hash.
    
    Args:
        filepath: Path to file
        expected_hash: Expected hash value
        
    Returns:
        True if hash matches, False otherwise
    """
    try:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        return file_hash == expected_hash
        
    except Exception as e:
        print(f"Error verifying file: {e}")
        return False


def main():
    """Main download function."""
    print("Rush AI Admissions Model Downloader")
    print("=" * 50)
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(exist_ok=True)
    
    model_path = MODELS_DIR / MODEL_NAME
    
    # Check if model already exists
    if model_path.exists():
        print(f"Model file already exists: {model_path}")
        
        # Verify integrity
        if verify_file(model_path, MODEL_HASH):
            print("✓ Model integrity verified")
            return 0
        else:
            print("✗ Model integrity check failed")
            response = input("Re-download model? (y/n): ")
            if response.lower() != 'y':
                return 1
    
    # Download model
    print(f"\nDownloading model from: {MODEL_URL}")
    print(f"Size: ~{MODEL_SIZE / 1024 / 1024:.0f} MB")
    print(f"Destination: {model_path}")
    
    if download_file(MODEL_URL, model_path, MODEL_SIZE):
        print("\n✓ Download complete")
        
        # Verify integrity
        if verify_file(model_path, MODEL_HASH):
            print("✓ Model integrity verified")
            print("\nModel ready to use!")
            return 0
        else:
            print("✗ Model integrity check failed")
            print("Please try downloading again or contact support")
            return 1
    else:
        print("\n✗ Download failed")
        print("\nAlternative: Please manually download the model and place it in:")
        print(f"  {MODELS_DIR}")
        return 1


if __name__ == "__main__":
    sys.exit(main())