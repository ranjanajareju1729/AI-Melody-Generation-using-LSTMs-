import tensorflow.keras as keras
import json
import numpy as np
import music21 as m21
import os
from tensorflow.keras.utils import to_categorical
from preprocessing import SEQUENCE_LENGTH, MAPPING_PATH, SINGLE_FILE_TEMPLATE

class MelodyGenerator:
    def __init__(self, model_path="melody_generator.keras"):
       
        with open(MAPPING_PATH) as fp:
            self._mapping = json.load(fp)
        
        # Load trained model with compatibility check
        self.model = keras.models.load_model(model_path)
        self._verify_model_compatibility()
        
        # Create reverse mapping and set sequence length
        self._reverse_mapping = {v: k for k, v in self._mapping.items()}
        self._sequence_length = SEQUENCE_LENGTH

    def _verify_model_compatibility(self):
        """Ensure model output matches vocabulary size"""
        output_shape = self.model.layers[-1].output.shape[-1]
        if output_shape != len(self._mapping):
            raise ValueError(
                f"Model expects {output_shape} outputs but mapping has {len(self._mapping)} symbols\n"
                "Retrain model with current preprocessing!"
            )

    def generate_melody(self, seed, num_steps=500, temperature=0.7):
        seed_symbols = [s for s in seed.split() if s in self._mapping]
        seed_symbols = self._pad_or_truncate_seed(seed_symbols)
        
        # Convert to numerical indices
        seed_indices = [self._mapping[s] for s in seed_symbols]
        melody_indices = seed_indices.copy()

        # Generate new notes
        for _ in range(num_steps):
            input_seq = np.array([seed_indices[-self._sequence_length:]])
            probabilities = self.model.predict(input_seq, verbose=0)[0]
            next_index = self._sample_with_temperature(probabilities, temperature)
            
            seed_indices.append(next_index)
            melody_indices.append(next_index)
            
            if self._reverse_mapping.get(next_index) == "/":  # Stop token
                break

        return [self._reverse_mapping.get(i, "r") for i in melody_indices]

    def _pad_or_truncate_seed(self, seed):
        if len(seed) < self._sequence_length:
            return ["r"] * (self._sequence_length - len(seed)) + seed
        return seed[-self._sequence_length:]

    def _sample_with_temperature(self, probs, temperature):
        probs = np.log(probs + 1e-10) / temperature  # Prevent log(0)
        exp_probs = np.exp(probs - np.max(probs))    # Prevent overflow
        return np.random.choice(len(exp_probs), p=exp_probs/np.sum(exp_probs))

    def save_melody(self, melody, filename="melody.mid", step_duration=0.25):
        #Convert symbol sequence to MIDI with musical details
        stream = m21.stream.Stream()
        current_note = None
        step_counter = 1

        # Add tempo metadata
        stream.insert(0, m21.tempo.MetronomeMark(number=120))

        for i, symbol in enumerate(melody):
            if symbol != "_" or i == len(melody)-1:
                if current_note is not None:
                    duration = step_duration * step_counter
                    self._add_note_to_stream(stream, current_note, duration)
                current_note = symbol
                step_counter = 1
            else:
                step_counter += 1

        stream.write("midi", filename)

    def _add_note_to_stream(self, stream, symbol, duration):
        #create notes/rests with error handling
        try:
            if symbol == "r":
                stream.append(m21.note.Rest(quarterLength=duration))
            else:
                note = m21.note.Note(
                    int(symbol),
                    quarterLength=duration
                )
                note.volume.velocity = 90  # Medium loudness
                stream.append(note)
        except:
            print(f"Invalid symbol {symbol}, using rest instead")
            stream.append(m21.note.Rest(quarterLength=duration))

    def evaluate_performance(self, split="val", batch_size=64):
        """Evaluate model on validation/test split"""
        with open(SINGLE_FILE_TEMPLATE.format(split)) as fp:
            sequences = fp.read().split(" / ")
        
        inputs, targets = [], []
        for seq in sequences:
            symbols = seq.split()
            inputs.append([self._mapping[s] for s in symbols[:-1]])
            targets.append(self._mapping[symbols[-1]])
        
        loss, acc = self.model.evaluate(
            np.array(inputs),
            to_categorical(targets, num_classes=len(self._mapping)),
            batch_size=batch_size,
            verbose=0
        )
        return {"loss": round(loss, 4), "accuracy": round(acc, 4)}

if __name__ == "__main__":

    generator = MelodyGenerator()
    
    # Quantitative evaluation
    metrics = generator.evaluate_performance("val")
    print(f"Validation Results - Loss: {metrics['loss']}, Accuracy: {metrics['accuracy']:.2%}")
    
    # Melody generation
    seed_phrase = "64 _ 62 _ 60 _ _ _ 62"  # MIDI pitches with sustains
    generated_melody = generator.generate_melody(seed_phrase, temperature=0.7)

    generator.save_melody(
        generated_melody,
        filename="generated_melody.mid",
        step_duration=0.25
    )
    print("Melody generated successfully!")