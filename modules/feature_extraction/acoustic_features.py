"""
acoustic_features.py
--------------------
Updated to support both .wav and .mp3 files.

Extracts:
- MFCCs
- Pitch
- Speech Rate
"""

import librosa
import numpy as np
from pydub import AudioSegment
import io

def load_audio(audio_path, target_sr=16000):
    """
    Loads an audio file (.wav or .mp3) and converts to a waveform.
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    
    Returns:
    --------
    Tuple (waveform, sample_rate)
    """
    if audio_path.endswith(".mp3"):
        audio = AudioSegment.from_mp3(audio_path)
        audio = audio.set_channels(1).set_frame_rate(target_sr)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        return samples / np.max(np.abs(samples)), target_sr
    else:
        return librosa.load(audio_path, sr=target_sr)

def extract_acoustic_features(audio_path):
    """
    Extracts MFCCs, Pitch, and Speech Rate from an audio file.
    Supports .mp3 and .wav formats.
    """
    y, sr = load_audio(audio_path)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

    # Speech Rate
    speech_rate = len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y) / sr)

    return {
        "MFCC_Mean": np.mean(mfccs),
        "Pitch": avg_pitch,
        "Speech Rate": speech_rate
    }
