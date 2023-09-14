import numpy as np
import h5py
import torch
import config as cfg
from train_valid_and_test import train_valid_model,test_model


def from_mat_to_tensor(raw_data):
    #transpose, the dimention of mat and numpy is contrary
    Transpose = np.transpose(raw_data)
    Nparray = np.array(Transpose)
    return Nparray

# all the number of sbjects in the experiment
# train one model for every subject

# read the data
eegname = cfg.process_data_dir + '/' +  cfg.dataset_name
eegdata = h5py.File(eegname, 'r')
data = from_mat_to_tensor(eegdata['EEG'])  # eeg data
label = from_mat_to_tensor(eegdata['ENV'])  # 0 or 1, representing the attended direction

# random seed
torch.manual_seed(2024)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(2024)

res = torch.zeros((cfg.sbnum,cfg.kfold_num))


from sklearn.model_selection import KFold,train_test_split
kfold = KFold(n_splits=cfg.kfold_num, shuffle=True, random_state=2024)

for sb in range(cfg.sbnum):
    # get the data of specific subject
    eegdata = data[sb]
    eeglabel = label[sb]

    eegdata = eegdata.reshape(8 * int(360 * 128 / cfg.decision_window), cfg.decision_window, 10, 11)
    eeglabel = eeglabel.reshape(8 * int(360 * 128 / cfg.decision_window), cfg.decision_window)


    for fold, (train_ids,  test_ids) in enumerate(kfold.split(eegdata)):
        train_valid_model(eegdata[train_ids], eeglabel[train_ids], sb, fold)
        res[sb,fold] = test_model(eegdata[test_ids], eeglabel[test_ids], sb,fold)
    print("good job!")

for sb in range(cfg.sbnum):
    print(sb)
    print(torch.mean(res[sb]))

np.savetxt('result.csv', res.numpy(), delimiter=',')


