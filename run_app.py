#!/usr/bin/env python3
"""
Brain Tumor Prediction System - Startup Script
Simple script to launch the medical UI application
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'torch',
        'sklearn',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_model_file():
    """Check if the trained model file exists"""
    model_path = "multi_output_brain_tumor_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("   Please ensure the trained model file is in the current directory")
        return False
    return True

def main():
    """Main startup function"""
    print("🧠 Brain Tumor Prediction System")
    print("=" * 40)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies found")
    
    # Check model file
    print("🔍 Checking model file...")
    if not check_model_file():
        sys.exit(1)
    print("✅ Model file found")
    
    # Launch Streamlit app
    print("\n🚀 Starting Brain Tumor Prediction System...")
    print("   The application will open in your default web browser")
    print("   Press Ctrl+C to stop the application")
    print("=" * 40)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
