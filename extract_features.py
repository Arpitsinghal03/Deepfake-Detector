import numpy as np
import librosa
from typing import Dict, List, Union
from scipy import signal

def extract_audio_features(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extract audio features for voice clone detection.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        
    Returns:
        np.ndarray: Extracted features
    """
    # Initialize feature list
    features = []
    
    # 1. Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features.extend([
        np.mean(mfccs, axis=1),  # Mean MFCCs
        np.std(mfccs, axis=1),   # Std MFCCs
        np.max(mfccs, axis=1),   # Max MFCCs
        np.min(mfccs, axis=1)    # Min MFCCs
    ])
    
    # 2. Extract spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    
    features.extend([
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth),
        np.mean(spectral_rolloff),
        np.std(spectral_rolloff)
    ])
    
    # 3. Extract chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend([
        np.mean(chroma, axis=1),  # Mean chroma
        np.std(chroma, axis=1)    # Std chroma
    ])
    
    # 4. Extract temporal features
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    rms_energy = librosa.feature.rms(y=audio)
    
    features.extend([
        np.mean(zero_crossing_rate),
        np.std(zero_crossing_rate),
        np.mean(rms_energy),
        np.std(rms_energy)
    ])
    
    # 5. Extract pitch features
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_mean = np.mean(pitches[magnitudes > np.median(magnitudes)])
    pitch_std = np.std(pitches[magnitudes > np.median(magnitudes)])
    
    features.extend([pitch_mean, pitch_std])
    
    # 6. Extract formant features
    formants = extract_formants(audio, sr)
    features.extend(formants)
    
    # 7. Extract prosody features
    prosody = extract_prosody_features(audio, sr)
    features.extend(prosody)
    
    # Convert to numpy array and normalize
    features = np.concatenate([f.flatten() for f in features])
    features = normalize_features(features)
    
    return features

def extract_formants(audio: np.ndarray, sr: int) -> List[float]:
    """
    Extract formant frequencies from audio signal.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        
    Returns:
        List[float]: Formant frequencies
    """
    # Apply pre-emphasis filter
    pre_emphasis = 0.97
    emphasized_audio = np.append(
        audio[0],
        audio[1:] - pre_emphasis * audio[:-1]
    )
    
    # Frame the signal
    frame_length = int(0.025 * sr)  # 25ms frames
    frame_step = int(0.010 * sr)    # 10ms step
    frames = librosa.util.frame(
        emphasized_audio,
        frame_length=frame_length,
        hop_length=frame_step
    )
    
    # Apply window function
    window = np.hamming(frame_length)
    windowed_frames = frames * window[:, np.newaxis]
    
    # Get formants for each frame
    formants = []
    for frame in windowed_frames.T:
        # Get LPC coefficients
        lpc_coeffs = librosa.lpc(frame, order=8)
        
        # Find roots of LPC polynomial
        roots = np.roots(lpc_coeffs)
        roots = roots[np.imag(roots) >= 0]  # Keep only upper half
        
        # Calculate formant frequencies
        angles = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angles * (sr / (2 * np.pi))
        
        # Sort by frequency and take first 3 formants
        freqs = np.sort(freqs)
        formants.extend(freqs[:3])
    
    # Calculate statistics of formants
    formant_stats = []
    for i in range(3):
        formant_freqs = [f[i] for f in formants if not np.isnan(f[i])]
        if formant_freqs:
            formant_stats.extend([
                np.mean(formant_freqs),
                np.std(formant_freqs)
            ])
        else:
            formant_stats.extend([0, 0])
    
    return formant_stats

def extract_prosody_features(audio: np.ndarray, sr: int) -> List[float]:
    """
    Extract prosody features from audio signal.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        
    Returns:
        List[float]: Prosody features
    """
    # Calculate energy envelope
    frame_length = int(0.025 * sr)
    frame_step = int(0.010 * sr)
    frames = librosa.util.frame(
        audio,
        frame_length=frame_length,
        hop_length=frame_step
    )
    energy = np.sum(frames ** 2, axis=0)
    
    # Calculate pitch contour
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch_contour = pitches[magnitudes > np.median(magnitudes)]
    
    # Extract features
    features = []
    
    # 1. Energy features
    features.extend([
        np.mean(energy),
        np.std(energy),
        np.max(energy) / (np.mean(energy) + 1e-6),
        np.min(energy) / (np.mean(energy) + 1e-6)
    ])
    
    # 2. Pitch features
    if len(pitch_contour) > 0:
        features.extend([
            np.mean(pitch_contour),
            np.std(pitch_contour),
            np.max(pitch_contour) / (np.mean(pitch_contour) + 1e-6),
            np.min(pitch_contour) / (np.mean(pitch_contour) + 1e-6)
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    # 3. Duration features
    duration = len(audio) / sr
    features.append(duration)
    
    # 4. Silence ratio
    silence_threshold = 0.01
    silence_ratio = np.mean(np.abs(audio) < silence_threshold)
    features.append(silence_ratio)
    
    return features

def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        features: Input features
        
    Returns:
        np.ndarray: Normalized features
    """
    # Replace inf and nan with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize to [0, 1] range
    features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-6)
    
    return features 