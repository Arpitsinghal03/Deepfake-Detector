import os
from pathlib import Path
from typing import Union, List

def validate_file(file_path: Union[str, Path]) -> bool:
    """
    Validate if a file exists and has a supported format.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    # Check if file exists
    if not file_path.is_file():
        return False
    
    # Check file extension
    supported_video = {'.mp4', '.avi', '.mov'}
    supported_audio = {'.wav', '.mp3'}
    supported_formats = supported_video | supported_audio
    
    return file_path.suffix.lower() in supported_formats

def get_file_type(file_path: Union[str, Path]) -> str:
    """
    Determine if a file is video or audio based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: 'video', 'audio', or 'unknown'
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    video_extensions = {'.mp4', '.avi', '.mov'}
    audio_extensions = {'.wav', '.mp3'}
    
    ext = file_path.suffix.lower()
    if ext in video_extensions:
        return 'video'
    elif ext in audio_extensions:
        return 'audio'
    return 'unknown'

def create_output_dir(base_dir: Union[str, Path], subdir: str = None) -> Path:
    """
    Create an output directory for processed files.
    
    Args:
        base_dir: Base directory path
        subdir: Optional subdirectory name
        
    Returns:
        Path: Path to the created directory
    """
    if not isinstance(base_dir, Path):
        base_dir = Path(base_dir)
    
    output_dir = base_dir / 'output'
    if subdir:
        output_dir = output_dir / subdir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_temp_dir() -> Path:
    """
    Get or create a temporary directory for processing files.
    
    Returns:
        Path: Path to the temporary directory
    """
    temp_dir = Path('temp')
    temp_dir.mkdir(exist_ok=True)
    return temp_dir

def cleanup_temp_files():
    """Remove all files in the temporary directory."""
    temp_dir = get_temp_dir()
    for file in temp_dir.glob('*'):
        if file.is_file():
            file.unlink()
    temp_dir.rmdir() 