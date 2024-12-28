import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from dataset import RFSignalDataset, load_data
from model import RFSignalClassifier
import torch.nn as nn
import numpy as np

def train_model():
    processed_dir = "data/processed"
    train_x, train_y, val_x, val_y, modtypes = load_data(processed_dir)

    # Convert one-hot labels to integer indices
    train_y = np.argmax(train_y, axis=1)
    val_y = np.argmax(val_y, axis=1)

    # Transpose data to match PyTorch's expectations
    train_x = train_x.T
    val_x = val_x.T

    train_dataset = RFSignalDataset(train_x, train_y)
    val_dataset = RFSignalDataset(val_x, val_y)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)

    # Set device to MPS for M1 Mac
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = RFSignalClassifier(input_size=train_x.shape[1], num_classes=len(modtypes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    early_stop_patience = 3
    no_improve_epochs = 0

    for epoch in range(10):
        model.train()
        epoch_loss = 0.0
        for signals, labels in train_loader:
            optimizer.zero_grad()
            signals = signals.permute(0, 2, 1).to(device)
            labels = labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.permute(0, 2, 1).to(device)
                labels = labels.to(device)
                outputs = model(signals)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_acc = 100 * correct / total

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), "best_model_mps.pth")
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
        else:
            no_improve_epochs += 1

        # Early stopping
        if no_improve_epochs >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        print(f"Epoch {epoch + 1}/{10}")
        print(f"    Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"    Validation Accuracy: {val_acc:.2f}%")

    print("Training complete. Best Validation Accuracy: {:.2f}%".format(best_acc))

if __name__ == "__main__":
    train_model()