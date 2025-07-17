import os
from pathlib import Path
from typing import Tuple
import moviepy.editor as mp
from .file_utils import get_temp_dir, create_output_dir

def split_video_audio(video_path: str) -> Tuple[str, str]:
    """
    Split a video file into separate video and audio files.
    
    Args:
        video_path: Path to the input video file
        
    Returns:
        Tuple[str, str]: Paths to the separated video and audio files
    """
    if not isinstance(video_path, Path):
        video_path = Path(video_path)
    
    # Create temporary directory for processing
    temp_dir = get_temp_dir()
    
    # Load the video
    video = mp.VideoFileClip(str(video_path))
    
    # Extract audio
    audio_path = temp_dir / f"{video_path.stem}_audio.wav"
    video.audio.write_audiofile(str(audio_path))
    
    # Save video without audio
    video_path_no_audio = temp_dir / f"{video_path.stem}_no_audio{video_path.suffix}"
    video.without_audio().write_videofile(str(video_path_no_audio))
    
    # Close the video to free up resources
    video.close()
    
    return str(video_path_no_audio), str(audio_path)

def extract_audio_from_video(video_path: str, output_dir: str = None) -> str:
    """
    Extract audio from a video file.
    
    Args:
        video_path: Path to the input video file
        output_dir: Optional output directory for the audio file
        
    Returns:
        str: Path to the extracted audio file
    """
    if not isinstance(video_path, Path):
        video_path = Path(video_path)
    
    # Determine output directory
    if output_dir:
        output_path = create_output_dir(output_dir, 'audio')
    else:
        output_path = get_temp_dir()
    
    # Extract audio
    audio_path = output_path / f"{video_path.stem}.wav"
    video = mp.VideoFileClip(str(video_path))
    video.audio.write_audiofile(str(audio_path))
    video.close()
    
    return str(audio_path)

def combine_video_audio(video_path: str, audio_path: str, output_dir: str = None) -> str:
    """
    Combine a video file with an audio file.
    
    Args:
        video_path: Path to the input video file
        audio_path: Path to the input audio file
        output_dir: Optional output directory for the combined file
        
    Returns:
        str: Path to the combined video file
    """
    if not isinstance(video_path, Path):
        video_path = Path(video_path)
    if not isinstance(audio_path, Path):
        audio_path = Path(audio_path)
    
    # Determine output directory
    if output_dir:
        output_path = create_output_dir(output_dir, 'combined')
    else:
        output_path = get_temp_dir()
    
    # Combine video and audio
    output_file = output_path / f"{video_path.stem}_with_audio{video_path.suffix}"
    video = mp.VideoFileClip(str(video_path))
    audio = mp.AudioFileClip(str(audio_path))
    
    # Ensure audio duration matches video
    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)
    elif audio.duration < video.duration:
        video = video.subclip(0, audio.duration)
    
    # Combine and write
    final_video = video.set_audio(audio)
    final_video.write_videofile(str(output_file))
    
    # Clean up
    video.close()
    audio.close()
    final_video.close()
    
    return str(output_file) 