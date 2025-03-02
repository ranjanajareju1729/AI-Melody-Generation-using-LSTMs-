import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import music21 as m21
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


KERN_DATASET_PATH = "split_dataset"
SAVE_DIR = "processed_dataset"
SPLITS = ["train", "val", "test"]
SINGLE_FILE_TEMPLATE = "single_file_{}.txt"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64
BATCH_SIZE = 128
ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]

def load_songs(dataset_path):
   
    songs = []
    for path, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".krn"):
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

def transpose_song(song):
    #Transpose song to C major/A minor
    key = song.analyze("key")
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    return song.transpose(interval)

def encode_song(song, time_step=0.25):
    #Convert music21 stream to encoded symbol sequence
    encoded = []
    for event in song.flatten().notesAndRests:
        if event.duration.quarterLength not in ACCEPTABLE_DURATIONS:
            raise ValueError(f"Invalid duration: {event.duration.quarterLength}")
        symbol = event.pitch.midi if isinstance(event, m21.note.Note) else "r"
        steps = int(event.duration.quarterLength / time_step)
        encoded.extend([symbol] + ["_"]*(steps-1))
    return " ".join(map(str, encoded))

def create_dataset():
    
    all_symbols = set()
    
    # First pass: collect all symbols
    for split in SPLITS:
        split_path = os.path.join(KERN_DATASET_PATH, split)
        songs = load_songs(split_path)
        
        for song in songs:
            try:
                song = transpose_song(song)
                encoded = encode_song(song)
                all_symbols.update(encoded.split())
            except Exception as e:
                print(f"Skipping song: {str(e)}")
                continue
    
    # Create global mapping
    mappings = {sym:i for i, sym in enumerate(sorted(all_symbols))}
    with open(MAPPING_PATH, "w") as fp:
        json.dump(mappings, fp, indent=4)
    
    # Second pass: process files with consistent mapping
    for split in SPLITS:
        split_path = os.path.join(KERN_DATASET_PATH, split)
        save_dir = os.path.join(SAVE_DIR, split)
        os.makedirs(save_dir, exist_ok=True)
        
        songs = load_songs(split_path)
        valid_songs = []
        
        for i, song in enumerate(songs):
            try:
                song = transpose_song(song)
                encoded = encode_song(song)
                with open(os.path.join(save_dir, f"{i}.txt"), "w") as fp:
                    fp.write(encoded)
                valid_songs.append(encoded)
            except Exception as e:
                print(f"Skipping {split} song {i}: {str(e)}")
        
        # Create single file per split
        sequences = []
        for song in valid_songs:
            symbols = song.split()
            for i in range(len(symbols) - SEQUENCE_LENGTH):
                sequences.append(" ".join(symbols[i:i+SEQUENCE_LENGTH+1]))
        
        with open(SINGLE_FILE_TEMPLATE.format(split), "w") as fp:
            fp.write(" / ".join(sequences))
    
    return len(mappings)

def data_generator(split="train", batch_size=BATCH_SIZE):
    """Generate batches from preprocessed data"""
    with open(SINGLE_FILE_TEMPLATE.format(split), "r") as fp:
        sequences = fp.read().split(" / ")
    
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
    
    vocab_size = len(mappings)
    
    while True:
        batch_indices = np.random.choice(len(sequences), size=batch_size)
        inputs, targets = [], []
        
        for idx in batch_indices:
            symbols = sequences[idx].split()
            inputs.append([mappings[sym] for sym in symbols[:-1]])
            targets.append(mappings[symbols[-1]])
        
        yield (
            np.array(inputs),
            to_categorical(targets, num_classes=vocab_size)
        )

if __name__ == "__main__":
    vocab_size = create_dataset()
    print(f"Vocabulary size: {vocab_size}")