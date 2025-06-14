import argparse
import streamlit as st
import os
from pathlib import Path

# Import local modules
from video_detector.detect_video import analyze_video
from audio_detector.detect_audio import analyze_audio
from utils.video_audio_splitter import split_video_audio
from utils.file_utils import validate_file

def parse_args():
    parser = argparse.ArgumentParser(description='Deepfake & Voice Detection Tool')
    parser.add_argument('--input', type=str, help='Path to input media file')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--mode', type=str, choices=['cli', 'web'], default='cli',
                      help='Interface mode: cli or web (default: cli)')
    return parser.parse_args()

def cli_interface(args):
    """Command Line Interface"""
    if args.input:
        # Handle combined video/audio file
        if not validate_file(args.input):
            print(f"Error: Invalid file {args.input}")
            return
        
        # Split video and audio if it's a video file
        if args.input.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path, audio_path = split_video_audio(args.input)
            video_result = analyze_video(video_path)
            audio_result = analyze_audio(audio_path)
            print_results(video_result, audio_result)
        else:
            print("Error: Unsupported file format")
    
    elif args.video:
        if not validate_file(args.video):
            print(f"Error: Invalid video file {args.video}")
            return
        result = analyze_video(args.video)
        print_results(video_result=result)
    
    elif args.audio:
        if not validate_file(args.audio):
            print(f"Error: Invalid audio file {args.audio}")
            return
        result = analyze_audio(args.audio)
        print_results(audio_result=result)
    
    else:
        print("Error: Please provide an input file")

def print_results(video_result=None, audio_result=None):
    """Print analysis results"""
    if video_result:
        print("\n=== Video Analysis Results ===")
        print(f"Deepfake Probability: {video_result['deepfake_probability']:.2%}")
        print(f"Confidence Score: {video_result['confidence_score']:.2%}")
        if 'face_analysis' in video_result:
            print("\nFace Analysis:")
            for face in video_result['face_analysis']:
                print(f"- Face {face['id']}: {face['manipulation_score']:.2%} manipulation detected")
    
    if audio_result:
        print("\n=== Audio Analysis Results ===")
        print(f"Voice Clone Probability: {audio_result['clone_probability']:.2%}")
        print(f"Confidence Score: {audio_result['confidence_score']:.2%}")
        if 'audio_features' in audio_result:
            print("\nAudio Features:")
            for feature, value in audio_result['audio_features'].items():
                print(f"- {feature}: {value}")

def web_interface():
    """Streamlit Web Interface"""
    st.title("Deepfake & Voice Detection Tool")
    
    st.sidebar.header("Upload Media")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a video or audio file",
        type=['mp4', 'avi', 'mov', 'wav', 'mp3']
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = Path("temp") / uploaded_file.name
        temp_path.parent.mkdir(exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process based on file type
        if uploaded_file.name.lower().endswith(('.mp4', '.avi', '.mov')):
            with st.spinner('Analyzing video...'):
                video_result = analyze_video(str(temp_path))
                st.subheader("Video Analysis Results")
                st.progress(video_result['deepfake_probability'])
                st.write(f"Deepfake Probability: {video_result['deepfake_probability']:.2%}")
                st.write(f"Confidence Score: {video_result['confidence_score']:.2%}")
                
                if 'face_analysis' in video_result:
                    st.subheader("Face Analysis")
                    for face in video_result['face_analysis']:
                        st.write(f"Face {face['id']}: {face['manipulation_score']:.2%} manipulation detected")
        
        elif uploaded_file.name.lower().endswith(('.wav', '.mp3')):
            with st.spinner('Analyzing audio...'):
                audio_result = analyze_audio(str(temp_path))
                st.subheader("Audio Analysis Results")
                st.progress(audio_result['clone_probability'])
                st.write(f"Voice Clone Probability: {audio_result['clone_probability']:.2%}")
                st.write(f"Confidence Score: {audio_result['confidence_score']:.2%}")
        
        # Clean up temporary file
        temp_path.unlink()
        temp_path.parent.rmdir()

def main():
    args = parse_args()
    
    if args.mode == 'web':
        web_interface()
    else:
        cli_interface(args)

if __name__ == "__main__":
    main() 