import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    if sys.version_info >= (3, 13):
        print("Warning: Python 3.13 is very new and some packages may not have pre-built wheels available.")
        print("You may need to build some packages from source or use alternative installation methods.")
        print("Consider using Python 3.8-3.11 for better compatibility.")

def check_ffmpeg():
    """Check if FFmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: FFmpeg is not installed or not in PATH")
        print("Please install FFmpeg from https://ffmpeg.org/download.html")
        sys.exit(1)

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print("Installing PyTorch...")
    try:
        # Check for CUDA
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except ImportError:
            pass

        if cuda_available:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ], check=True)
        else:
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ], check=True)
    except subprocess.SubprocessError as e:
        print(f"Error installing PyTorch: {e}")
        sys.exit(1)

def install_dlib():
    """Install dlib using pre-built wheel or alternative methods"""
    print("Installing dlib...")
    try:
        python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
        
        if platform.system() == "Windows":
            # For Python 3.13, we'll try alternative methods
            if sys.version_info >= (3, 13):
                print("Python 3.13 detected. Attempting alternative installation methods...")
                try:
                    # First try installing from PyPI
                    print("Attempting to install dlib from PyPI...")
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", "dlib"
                    ], check=True)
                except subprocess.SubprocessError:
                    print("\nFailed to install dlib from PyPI. Please try one of these alternatives:")
                    print("1. Install Visual Studio Build Tools and build dlib from source:")
                    print("   - Download Visual Studio Build Tools from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
                    print("   - Install with C++ build tools")
                    print("   - Then run: pip install dlib")
                    print("2. Use a pre-built wheel from an alternative source:")
                    print("   - Visit: https://github.com/z-mahmud22/Dlib_Windows_Python")
                    print("   - Download the appropriate wheel for your system")
                    print("   - Install using: pip install <downloaded-wheel-file>")
                    print("3. Consider using Python 3.8-3.11 which has better package support")
                    sys.exit(1)
            else:
                # For Python 3.8-3.11, use the pre-built wheel
                wheel_url = f"https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-{python_version}-{python_version}-win_amd64.whl"
                try:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", wheel_url
                    ], check=True)
                except subprocess.SubprocessError:
                    print(f"\nFailed to install dlib from {wheel_url}")
                    print("Attempting to install from PyPI...")
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", "dlib"
                    ], check=True)
        else:
            # For non-Windows systems, try to build from source
            subprocess.run([
                sys.executable, "-m", "pip", "install", "dlib"
            ], check=True)
    except subprocess.SubprocessError as e:
        print(f"Error installing dlib: {e}")
        print("\nIf you're using Python 3.13, please consider:")
        print("1. Using Python 3.8-3.11 for better package support")
        print("2. Installing Visual Studio Build Tools and building dlib from source")
        print("3. Using a pre-built wheel from an alternative source")
        sys.exit(1)

def install_pyav():
    """Install PyAV with fallback options"""
    print("Installing PyAV...")
    try:
        # First try with pre-built binary
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "av", "--no-binary", "av", "--prefer-binary"
        ], check=True)
    except subprocess.SubprocessError:
        try:
            # Fallback to specific version
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "av==9.2.0"  # Known stable version
            ], check=True)
        except subprocess.SubprocessError as e:
            print(f"Error installing PyAV: {e}")
            print("Please try installing PyAV manually:")
            print("pip install av==9.2.0")
            sys.exit(1)

def install_requirements():
    """Install remaining requirements"""
    print("Installing remaining requirements...")
    try:
        requirements_file = Path(__file__).parent / "requirements.txt"
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
    except subprocess.SubprocessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def main():
    """Main installation function"""
    print("Starting installation process...")
    
    # Check Python version
    check_python_version()
    
    # Check FFmpeg
    check_ffmpeg()
    
    # Install dependencies in correct order
    install_pytorch()
    install_dlib()
    install_pyav()
    install_requirements()
    
    print("\nInstallation completed successfully!")
    print("\nTo verify the installation, you can run:")
    print("python -c 'import torch; import dlib; import av; print(\"All dependencies installed successfully!\")'")

if __name__ == "__main__":
    main() 