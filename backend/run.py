#!/usr/bin/env python3
"""
One-command backend startup script
Simple command: python run.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the backend server with one command"""
    print("🚀 Starting Stock Price API Backend")
    print("=" * 50)
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Check if we're in the right directory
    if not (script_dir / "main.py").exists():
        print("❌ Error: main.py not found. Make sure you're in the backend directory.")
        sys.exit(1)
    
    # Check if virtual environment exists
    venv_path = script_dir / "venv"
    if not venv_path.exists():
        print("❌ Virtual environment not found!")
        print("Please run: python -m venv venv")
        sys.exit(1)
    
    # Determine the Python executable path
    if os.name == 'nt':  # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
    else:  # macOS/Linux
        python_exe = venv_path / "bin" / "python"
    
    if not python_exe.exists():
        print(f"❌ Python executable not found at {python_exe}")
        sys.exit(1)
    
    print(f"✅ Using Python: {python_exe}")
    print(f"✅ Starting server from: {script_dir}")
    print("=" * 50)
    
    try:
        # Start the server
        print("🚀 Starting server...")
        subprocess.run([str(python_exe), "main.py"], cwd=script_dir, check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server exited with error code: {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
