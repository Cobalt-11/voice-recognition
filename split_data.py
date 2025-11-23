import os
import pandas as pd

# Define paths
home_dir = os.path.expanduser("~")
data_dir = os.path.join(home_dir, "Documents/voice-recognition/processed")
train_path = os.path.join(data_dir, "balanced_train.tsv")
valid_path = os.path.join(data_dir, "balanced_valid.tsv")

# Load datasets
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Training file not found: {train_path}")
if not os.path.exists(valid_path):
    raise FileNotFoundError(f"Validation file not found: {valid_path}")

train_df = pd.read_csv(train_path, delimiter='\t')
valid_df = pd.read_csv(valid_path, delimiter='\t')

# Display current sizes
print(f"Current sizes: Train = {len(train_df)}, Validation = {len(valid_df)}")

# Calculate the number of samples to move per demographic group (even split)
num_groups = valid_df.groupby(['gender', 'age_group']).size()
samples_per_group = 1000 // len(num_groups)  # Divide evenly across groups

# Initialize list for samples to move
samples_to_move = []

# Move samples from validation to training while maintaining demographics
for (gender, age_group), group in valid_df.groupby(['gender', 'age_group']):
    # Select the first `samples_per_group` samples from each demographic group
    selected_samples = group.iloc[:samples_per_group]
    samples_to_move.append(selected_samples)

# Combine all selected samples into a single DataFrame
samples_to_move_df = pd.concat(samples_to_move)

# Remove these samples from the validation set
remaining_validation_df = valid_df.drop(samples_to_move_df.index)

# Add these samples to the training set
updated_train_df = pd.concat([train_df, samples_to_move_df]).reset_index(drop=True)

# Save updated datasets
updated_train_path = os.path.join(data_dir, "updated_balanced_train.tsv")
remaining_valid_path = os.path.join(data_dir, "updated_balanced_validated.tsv")

updated_train_df.to_csv(updated_train_path, sep='\t', index=False)
remaining_validation_df.to_csv(remaining_valid_path, sep='\t', index=False)

# Display new sizes
print(f"New sizes: Train = {len(updated_train_df)}, Validation = {len(remaining_validation_df)}")

# Function to print demographic distribution in percentages
def print_demographics(df, dataset_name):
    """Print demographic distribution in percentages"""
    total_samples = len(df)
    if total_samples == 0:
        print(f"{dataset_name} dataset is empty!")
        return
    
    demographics = df.groupby(['gender', 'age_group']).size()
    print(f"\n{dataset_name} Demographics:")
    for (gender, age_group), count in demographics.items():
        gender_display = "Male" if gender == "male_masculine" else "Female"
        age_display = "Kid" if age_group == "kid" else "Adult"
        percentage = (count / total_samples) * 100
        print(f"  {gender_display} {age_display}: {count} samples ({percentage:.2f}%)")
    print(f"  Total samples: {total_samples}")

# Print updated demographics for both datasets
print_demographics(updated_train_df, "Updated Training")
print_demographics(remaining_validation_df, "Updated Validation")

print(f"\nUpdated datasets saved at:\n  Training: {updated_train_path}\n  Validation: {remaining_valid_path}")
