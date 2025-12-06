import jsonlines
import random
import os

generic_train = os.path.join(os.path.dirname(__file__), "dataset_generic_train.jsonl")
train = os.path.join(os.path.dirname(__file__), "dataset_train.jsonl")

combined_train = []

# Load breed-specific data
with jsonlines.open(train) as reader:
    breed_train_data = list(reader)
    combined_train.extend(breed_train_data)

# Load generic data and oversample
with jsonlines.open(generic_train) as reader:
    generic_train_data = list(reader)
    generic_train_data *= 25  # (adjusted to get ~15% of dataset)
    combined_train.extend(generic_train_data)

# Shuffle
random.shuffle(combined_train)

# Save
with jsonlines.open("dataset_train.jsonl", "w") as writer:
    for ex in combined_train:
        writer.write(ex)
