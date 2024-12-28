import numpy as np
import torch
from torch.utils.data import Dataset

class RFSignalDataset(Dataset):
    def __init__(self, signals, labels, augment=False, target_classes=None):
        # Convert complex data to magnitude and phase
        magnitude = np.sqrt(np.real(signals)**2 + np.imag(signals)**2)
        phase = np.arctan2(np.imag(signals), np.real(signals))
        signals = np.stack([magnitude, phase], axis=-1)

        # Apply targeted augmentation
        if augment and target_classes:
            for target_class in target_classes:
                class_indices = np.where(labels == target_class)[0]
                for idx in class_indices:
                    # Add Gaussian noise to selected class samples
                    noise = np.random.normal(0, 0.01, signals[idx].shape)
                    signals[idx] += noise

                    # Time shift selected class samples
                    shift = np.random.randint(1, 20)
                    signals[idx] = np.roll(signals[idx], shift, axis=0)

        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

def load_data(processed_dir):
    train_x = np.load(f"{processed_dir}/train_x.npy")
    train_y = np.load(f"{processed_dir}/train_y.npy")
    val_x = np.load(f"{processed_dir}/val_x.npy")
    val_y = np.load(f"{processed_dir}/val_y.npy")
    modtypes = np.load(f"{processed_dir}/modtypes.npy", allow_pickle=True)

    # Transpose one-hot encoded labels to (samples, classes)
    if train_y.shape[0] == 11:  # Assumes 11 modulation classes
        train_y = train_y.T  # (11, 267300) -> (267300, 11)
        val_y = val_y.T      # (11, 29700) -> (29700, 11)

    print(f"Training data shape: {train_x.shape}, {train_y.shape}")
    print(f"Validation data shape: {val_x.shape}, {val_y.shape}")
    return train_x, train_y, val_x, val_y, modtypes