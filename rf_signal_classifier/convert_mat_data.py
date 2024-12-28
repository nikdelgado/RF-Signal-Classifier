from os.path import isfile, join
import os
import numpy as np
import mat73

def preprocess_data(mat_file_path, output_dir, val_split=0.1):
    """
    Load and preprocess data from a .mat file.
    
    Args:
        mat_file_path (str): Path to the MATLAB .mat file.
        output_dir (str): Directory to save processed data.
        val_split (float): Fraction of data to use for validation.
    """
    if not isfile(mat_file_path):
        raise FileNotFoundError(f"MAT file not found: {mat_file_path}")
    
    # Load the .mat file
    print(f"Loading data from {mat_file_path}...")
    matf = mat73.loadmat(mat_file_path)
    
    # Excise data from dictionary
    y_train = np.array(matf['rxTrainLabel'])
    X_train = np.array(matf['rxTrainData'])
    modtypes = matf['modulationTypes']
    
    # Create validation data from the training set
    Ns = X_train.shape
    Nv = int(np.round(Ns[1] * val_split))
    
    X_val = X_train[:, :Nv]
    X_train = X_train[:, Nv:]
    
    y_val = y_train[:, :Nv]
    y_train = y_train[:, Nv:]
    
    # Save data as numpy arrays
    print("Saving processed data...")
    os.makedirs(output_dir, exist_ok=True)
    np.save(join(output_dir, 'train_x.npy'), X_train)
    np.save(join(output_dir, 'train_y.npy'), y_train)
    np.save(join(output_dir, 'modtypes.npy'), modtypes)
    np.save(join(output_dir, 'val_x.npy'), X_val)
    np.save(join(output_dir, 'val_y.npy'), y_val)
    
    print("Data preprocessing complete. Files saved to:", output_dir)

if __name__ == "__main__":
    # Define paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    mat_file_path = join(project_root, "RF-Signal-Classifier", "data", "raw", "testData.mat")
    output_dir = join(project_root, "RF-Signal-Classifier", "data", "processed")
    
    # Run preprocessing
    preprocess_data(mat_file_path, output_dir)