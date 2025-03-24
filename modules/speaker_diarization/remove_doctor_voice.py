import os
import torch
import torchaudio
import librosa
import numpy as np
from torchaudio.transforms import Resample
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Segment


# Load the pre-trained speaker diarization model
model = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization")

def process_audio_file(audio_path, output_path):
    """Extracts only the patient’s voice and saves it to a new file."""
    
    # Load the audio file
    audio, sr = torchaudio.load(audio_path)
    audio = Resample(sr, 16000)(audio)  # Resample to 16kHz

    # Perform speaker diarization
    diarization = model({"uri": audio_path, "audio": audio_path})

    patient_segments = []
    
    # Extract patient speech (assuming SPEAKER_01 is the patient)
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker == "SPEAKER_01":  # Change if patient is labeled differently
            start, end = turn.start, turn.end
            patient_segments.append(audio[:, int(start * sr): int(end * sr)])

    # Save extracted speech if found
    if patient_segments:
        patient_audio = torch.cat(patient_segments, dim=1)  # Merge segments
        torchaudio.save(output_path, patient_audio, 16000)

    print(f"Processed: {audio_path} -> {output_path}")

def batch_process_audio():
    """Processes all audio files in the dataset and removes doctor’s voice."""
    
    INPUT_FOLDER = "../../data/raw_audio_files"
    OUTPUT_FOLDER = "../../data/processed_audio_files"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".wav")]

    for file in files:
        input_path = os.path.join(INPUT_FOLDER, file)
        output_path = os.path.join(OUTPUT_FOLDER, file)
        process_audio_file(input_path, output_path)

    print("All recordings processed successfully!")

if __name__ == "__main__":
    batch_process_audio()
