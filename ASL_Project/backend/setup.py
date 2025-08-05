#!/usr/bin/env python3
"""
Setup script for ASL Recognition Project
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True


def create_directories():
    """Create necessary directories"""
    directories = ["uploads"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f" Created directory: {directory}")

def check_model():
    """Check if model exists"""
    model_path = "model/asl_model.h5"
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        print(f"Model found: {model_path}")
        return True
    else:
        print(f"No trained model found at {model_path}")
        print("   You'll need to train the model or download a pre-trained one.")
        return False

def main():
    print(" Setting up ASL Recognition Project...")
    
    # Install requirements
    if not install_requirements():
        return
    
    # Create directories
    create_directories()
    
    # Check model
    check_model()
    
    print("\n Next steps:")
    print("1. If you don't have a trained model, you'll need to:")
    print("   - Use the asl_alphabet_train dataset")
    print("   - Train the model using the training script")
    print("2. Start the backend: python main.py")
    print("3. Open frontend/index.html in your browser")
    print("4. Test with sample ASL images")

if __name__ == "__main__":
    main() 