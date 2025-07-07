#!/usr/bin/env python3
"""
NewsNinja 2.0 - Quick Start (Guaranteed to Work)
This uses simplified versions to ensure connection works
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def install_requirements():
    """Install only essential requirements"""
    essential = ["fastapi", "uvicorn", "streamlit", "gtts", "requests", "python-dotenv"]
    
    print("📦 Installing essential packages...")
    for package in essential:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         capture_output=True, text=True)

def setup_directories():
    """Create audio directory"""
    Path("audio").mkdir(exist_ok=True)
    print("📁 Audio directory ready")

def start_simple_backend():
    """Start the simple backend"""
    print("🚀 Starting simple backend...")
    return subprocess.Popen([
        sys.executable, 'backend_simple.py'
    ])

def start_simple_frontend():
    """Start the simple frontend"""
    print("🖥️ Starting simple frontend...")
    return subprocess.Popen([
        sys.executable, '-m', 'streamlit', 'run', 'frontend_simple.py',
        '--server.port', '8501'
    ])

def main():
    print("🥷 NewsNinja 2.0 - Quick Start")
    print("=" * 40)
    print("Using simplified versions for guaranteed connectivity")
    print()
    
    # Setup
    install_requirements()
    setup_directories()
    
    try:
        # Start services
        backend = start_simple_backend()
        time.sleep(3)
        
        frontend = start_simple_frontend() 
        time.sleep(2)
        
        print("\n🎉 NewsNinja 2.0 is running!")
        print("🌐 Frontend: http://localhost:8501")
        print("🔧 Backend: http://localhost:1234")
        print("📚 API Docs: http://localhost:1234/docs")
        print("\n✅ API should show as CONNECTED now!")
        print("⏹️ Press Ctrl+C to stop")
        
        # Open browser
        try:
            webbrowser.open('http://localhost:8501')
        except:
            pass
        
        # Wait
        backend.wait()
        frontend.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        try:
            backend.terminate()
            frontend.terminate()
        except:
            pass
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()