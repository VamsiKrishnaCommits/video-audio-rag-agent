#!/usr/bin/env python3
"""
Quick setup verification script.
Run this after installing dependencies to verify everything is set up correctly.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version is 3.10+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required. Found: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check all required packages are installed"""
    required = {
        'google.genai': 'google-genai',
        'cv2': 'opencv-python',
        'pyaudio': 'pyaudio',
        'PIL': 'pillow',
        'mss': 'mss',
        'dotenv': 'python-dotenv',
        'certifi': 'certifi',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing.append(package)
    
    return len(missing) == 0

def check_env_file():
    """Check if .env file exists and has API key"""
    env_path = Path('app/.env')
    if not env_path.exists():
        print(f"❌ {env_path} not found. Copy from app/.env.example")
        return False
    
    with open(env_path) as f:
        content = f.read()
        if 'PASTE_YOUR_ACTUAL_API_KEY_HERE' in content or 'GEMINI_API_KEY=' not in content:
            print(f"⚠️  {env_path} exists but API key may not be set")
            return False
    
    print(f"✅ {env_path} found")
    return True

def check_ssl_cert():
    """Check SSL certificate path"""
    try:
        import certifi
        cert_path = certifi.where()
        if os.path.exists(cert_path):
            print(f"✅ SSL certificate found: {cert_path}")
            return True
        else:
            print(f"⚠️  SSL certificate path exists but file not found: {cert_path}")
            return False
    except Exception as e:
        print(f"⚠️  Could not verify SSL certificate: {e}")
        return False

def main():
    print("=" * 60)
    print("Setup Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    print("1. Python Version:")
    all_ok &= check_python_version()
    print()
    
    print("2. Dependencies:")
    all_ok &= check_dependencies()
    print()
    
    print("3. Environment File:")
    all_ok &= check_env_file()
    print()
    
    print("4. SSL Certificate:")
    check_ssl_cert()  # Warning only, not critical
    print()
    
    print("=" * 60)
    if all_ok:
        print("✅ Setup looks good! You can run the agent now.")
        print("\nTo start:")
        print("  cd app")
        print("  python google_search_agent/agent.py --knowledge-folder ../knowledge_docs")
    else:
        print("❌ Some issues found. Please fix them before running the agent.")
        sys.exit(1)

if __name__ == '__main__':
    main()
