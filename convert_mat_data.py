# Convert mat file variables to numpy arrays
# Uses mat73 library
#
# J. Shima

from os.path import isfile, join
import numpy as np
import mat73

# ---- replace with your own directory where mat data lives ----
root_dir = 'c:/projects/ai_challenge_mod_id/train_data' 
mypath = root_dir

# load up from matlab gen file
matf = mat73.loadmat(join(root_dir,'ai_rf_challenge_train_data.mat'))

#excise data from dict
y_train = np.array(matf['rxTrainLabel'])
X_train = np.array(matf['rxTrainData'])
modtypes = matf['modulationTypes']
    
#create validation data from training set, use 10% for validation
#data was already shuffled so just excise it straight up
val_per = 0.1
Ns = X_train.shape
Nv = np.round(Ns[1]*val_per)
Nv = Nv.astype(int)

#make val and new train data sets w/ labels
#note labels are one-hot encoded vectors
X_val = X_train[:,0:Nv]
X_train = X_train[:,Nv:]
    
y_val = y_train[:,0:Nv]
y_train = y_train[:,Nv:]
    
#save off as numpy arrays
print("Saving")
np.save(join(mypath,'train_x.npy'), X_train)
np.save(join(mypath,'train_y.npy'), y_train)
np.save(join(mypath,'modtypes.npy'), modtypes)
np.save(join(mypath,'val_x.npy'), X_val)
np.save(join(mypath,'val_y.npy'), y_val)

