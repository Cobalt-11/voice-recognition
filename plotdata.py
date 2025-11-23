import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk 
import librosa.display
from collections import Counter

# Učitavanje datasetova iz root foldera.
train_dataset = load_from_disk("processed_train_dataset")
#test_dataset = load_from_disk("processed_test_dataset")
validation_dataset = load_from_disk("processed_validation_dataset")

# Kod je namjenjen za prikaz dataset-ova i izgled signala
# Provjera samplova 
# Prikazuje samo prvi sample u baze.
train_sample_audio = train_dataset[0]["audio"]["array"]
plt.figure(figsize=(10,4))
plt.plot(train_sample_audio)
plt.title("Processed Audio Signal (Train)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()


val_sample_audio = validation_dataset[0]["audio"]["array"]
plt.figure(figsize=(10,4))
plt.plot(val_sample_audio)
plt.title("Processed Audio Signal (Val)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
"""
test_sample_audio = test_dataset[0]["audio"]["array"]
plt.figure(figsize=(10,4))
plt.plot(test_sample_audio)
plt.title("Processed Audio Signal (Test)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.show()
"""
"""
# Ovo je namjenjeno za prikaz mfcc uzoraka 

mfcc_sample = train_dataset[0]["mfcc"]
plt.figure(figsize=(10,4))
librosa.display.specshow(np.array(mfcc_sample).T, x_axis="time", sr=16000)
plt.colorbar()
plt.title("Mfcc")
"""    

total = len(train_dataset)
print(f"Total samples: {total}\n")

# Count gender and age
gender_counts = Counter(train_dataset['gender'])
age_counts = Counter(train_dataset['age'])

# Function to calculate and print percentages
def print_distribution(title, counter):
    print(f"{title} distribution:")
    for key, count in counter.items():
        label = key if key is not None else "undefined"
        percent = (count / total) * 100
        print(f"  {label:10}: {count:6} ({percent:.2f}%)")
    print()

# Print results
print_distribution("Gender", gender_counts)
print_distribution("Age", age_counts)