import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import soundfile as sf  # Replace librosa.output with soundfile

# Configuration
base_dir = os.path.expanduser("~/Documents/voice-recognition")
clips_dir = os.path.join(base_dir, "dataset/clips")
output_dir = os.path.join(base_dir, "processed_audio")
os.makedirs(output_dir, exist_ok=True)

# Paths to your balanced datasets
train_tsv = os.path.join(base_dir, "processed/updated_balanced_train.tsv")
valid_tsv = os.path.join(base_dir, "processed/updated_balanced_validated.tsv")

def process_and_copy_audio(tsv_path):
    """Process audio files from TSV and copy to new directory"""
    df = pd.read_csv(tsv_path, delimiter='\t')
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(tsv_path)}"):
        try:
            # Original audio path
            orig_path = os.path.join(clips_dir, row['path'])
            
            # New audio path (maintain directory structure)
            new_path = os.path.join(output_dir, row['path'])
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            # Load and process audio
            y, sr = librosa.load(orig_path, sr=16000)
            y_trimmed, _ = librosa.effects.trim(y, top_db=60)

            
            # Save processed audio using soundfile instead of librosa.output
            sf.write(new_path, y_trimmed, sr)  # Updated write command
            
            # Copy associated text
            txt_path = orig_path.replace('.wav', '.txt')
            if os.path.exists(txt_path):
                shutil.copy(txt_path, new_path.replace('.wav', '.txt'))
                
        except Exception as e:
            print(f"\nError processing {row['path']}: {str(e)}")
            continue

# Process both datasets
print("Starting audio processing...")
process_and_copy_audio(train_tsv)
process_and_copy_audio(valid_tsv)

print(f"\nProcessing complete! All audio files saved to: {output_dir}")
