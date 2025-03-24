"""
extract_features.py
-------------------
Extracts acoustic features from the Lu database audio files.

- Supports both `.wav` and `.mp3` formats
- Reads from two folders: control and dementia
- Adds labels to the extracted feature set
- Saves all features into `lu_acoustic_features.csv`

Author: Lavan Aditya
"""

from acoustic_features import extract_acoustic_features
import pandas as pd
import os

# Define paths
base_path = "../../data/lu_database"
control_folder = os.path.join(base_path, "processed_audio_control")
dementia_folder = os.path.join(base_path, "processed_audio_dementia")
output_file = "lu_acoustic_features.csv"

def process_folder(folder_path, label):
    """
    Extract features from all audio files in a folder and label them.
    
    Parameters:
    -----------
    folder_path : str
        Path to folder containing audio files.
    label : str
        'Control' or 'Dementia' for labeling the data.
    
    Returns:
    --------
    List[dict]
        List of feature dictionaries with labels.
    """
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".wav") or file.endswith(".mp3"):
            file_path = os.path.join(folder_path, file)
            features = extract_acoustic_features(file_path)
            features["Filename"] = file
            features["Label"] = label
            data.append(features)
    return data

def process_all():
    """
    Processes control and dementia audio files and saves to a CSV.
    """
    all_features = []
    all_features.extend(process_folder(control_folder, "Control"))
    all_features.extend(process_folder(dementia_folder, "Dementia"))

    df = pd.DataFrame(all_features)
    df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    process_all()
    print("All feature extraction completed successfully!")
