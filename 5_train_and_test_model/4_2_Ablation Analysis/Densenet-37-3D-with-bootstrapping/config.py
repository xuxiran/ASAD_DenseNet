import torch

# models
model_names = ['DenseNet_37-3D','DenseNet_37-I3D']
model_name = model_names[1] # you could change the code to other models by only changing the number
process_data_dir = '../../4_processed_data'
if model_name == 'CNN_baseline':
    dataset_name = 'KUL_1D.mat'
else:
    dataset_name = 'KUL_2D.mat'

device_ids = 2
device = torch.device(f"cuda:{device_ids}" if torch.cuda.is_available() else "cpu")
epoch_num = 10
batch_size = 128
sample_rate = 128
categorie_num = 2
sbnum = 16
kfold_num = 5

lr=1e-3
weight_decay=0.01

# the length of decision window
decision_window = 128 #1s



