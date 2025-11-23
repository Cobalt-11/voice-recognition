import os
import pandas as pd
import numpy as np

home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, "Documents/voice-recognition/dataset")
output_dir = os.path.join(home_dir, "Documents/voice-recognition/processed")

os.makedirs(output_dir, exist_ok=True)

def analyze_demographics(tsv_path, dataset_name):
    """Analyze demographics with percentage breakdowns"""
    print(f"\n===== {dataset_name} Dataset Demographics =====")
    
    chunk_size = 10000
    demographic_counts = {
        'male_masculine_kid': 0,
        'male_masculine_adult': 0,
        'female_feminine_kid': 0,
        'female_feminine_adult': 0
    }
    
    total_samples = 0
    valid_samples = 0
    
    try:
        for chunk in pd.read_csv(tsv_path, delimiter='\t', chunksize=chunk_size):
            total_samples += len(chunk)
            
            filtered = chunk[
                (chunk['gender'].isin(['male_masculine', 'female_feminine'])) & 
                (chunk['age'].isin(['teens', 'twenties', 'thirties', 'fourties', 
                                   'fifties', 'sixties', 'seventies', 'eighties', 'nineties']))
            ].copy()
            
            filtered.loc[:, 'age_group'] = np.where(
                filtered['age'] == 'teens', 'kid', 'adult'
            )
            
            valid_samples += len(filtered)
            
            for gender in ['male_masculine', 'female_feminine']:
                for age_group in ['kid', 'adult']:
                    mask = (filtered['gender'] == gender) & (filtered['age_group'] == age_group)
                    count = mask.sum()
                    key = f"{gender}_{age_group}"
                    demographic_counts[key] += count

        total_valid = sum(demographic_counts.values())
        print(f"Total samples: {total_samples}")
        print(f"Valid samples: {total_valid} ({total_valid/total_samples:.2%})")
        
        print("\nDemographic breakdown:")
        for key, count in demographic_counts.items():
            gender, age = key.split('_', 1)
            gender_display = gender.replace('masculine', '').replace('feminine', '').strip().capitalize()
            age_display = age.capitalize()
            print(f"  {gender_display} {age_display}: {count} ({(count/total_valid):.2%})")
            
    except FileNotFoundError:
        print(f"Error: File not found at {tsv_path}")
        return None
        
    return demographic_counts

def create_balanced_sample(tsv_path, output_path, target=500):
    """Create balanced dataset with proper key handling"""
    samples = {
        'male_masculine_kid': [],
        'male_masculine_adult': [],
        'female_feminine_kid': [],
        'female_feminine_adult': []
    }
    
    try:
        for chunk in pd.read_csv(tsv_path, delimiter='\t', chunksize=10000):
            filtered = chunk[
                (chunk['gender'].isin(['male_masculine', 'female_feminine'])) & 
                (chunk['age'].isin(['teens', 'twenties', 'thirties', 'fourties', 
                                   'fifties', 'sixties', 'seventies', 'eighties', 'nineties']))
            ].copy()
            
            filtered.loc[:, 'age_group'] = np.where(
                filtered['age'] == 'teens', 'kid', 'adult'
            )
            
            for _, row in filtered.iterrows():
                key = f"{row['gender']}_{row['age_group']}"
                if key in samples and len(samples[key]) < target:
                    samples[key].append(row.to_dict())
            
            if all(len(v) >= target for v in samples.values()):
                break

        balanced = pd.DataFrame([item for sublist in samples.values() for item in sublist])
        balanced.to_csv(output_path, sep='\t', index=False)
        return balanced
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {tsv_path}")
        return None

def main():
    train_tsv = os.path.join(data_dir, "train.tsv")
    valid_tsv = os.path.join(data_dir, "validated.tsv")
    balanced_train_path = os.path.join(output_dir, "balanced_train.tsv")
    balanced_valid_path = os.path.join(output_dir, "balanced_valid.tsv")

    missing_files = []
    for f in [train_tsv, valid_tsv]:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if missing_files:
        print("Missing required files:")
        for f in missing_files:
            print(f"  {f}")
        return

    print("Analyzing original datasets...")
    analyze_demographics(train_tsv, "Original Training")
    analyze_demographics(valid_tsv, "Original Validation")

    print("\nCreating balanced datasets...")
    train_balanced = create_balanced_sample(train_tsv, balanced_train_path)
    valid_balanced = create_balanced_sample(valid_tsv, balanced_valid_path)
    
    if train_balanced is None or valid_balanced is None:
        print("Failed to create balanced datasets")
        return

    print("\nAnalyzing balanced datasets...")
    analyze_demographics(balanced_train_path, "Balanced Training")
    analyze_demographics(balanced_valid_path, "Balanced Validation")

    print("\nProcessing complete!")
    print(f"Balanced datasets saved to:\n  {balanced_train_path}\n  {balanced_valid_path}")

if __name__ == "__main__":
    main()
