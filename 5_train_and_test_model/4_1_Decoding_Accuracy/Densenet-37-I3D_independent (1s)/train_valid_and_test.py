from AADdataset import AADdataset_1point,AADdataset_1second
from torch.utils.data import DataLoader
from model_2D import DenseNet_2D
from model_3D import DenseNet_3D
import tqdm
import torch
import config as cfg
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import os

writer = SummaryWriter()


# train the model for every subject
def train_valid_model(eegdata, eeglabel, fold):

# Train and test using the current folded data
    x_train_val, y_train_val = eegdata, eeglabel
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=2024)


# ----------------------initial model------------------------

    model_2D = DenseNet_2D().to(cfg.device)

    train_dataset = AADdataset_1point(x_train, y_train)
    valid_dataset = AADdataset_1point(x_valid, y_valid)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.batch_size, shuffle=True)

    # set the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_2D = torch.optim.AdamW(model_2D.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


# ---------------------if I3D, train and valid the model_2D-----------

    if cfg.model_name == 'DenseNet_37-I3D':
        valid_loss_min = 100
        for epoch in range(cfg.epoch_num):

            # train the model_2D
            num_correct = 0
            num_samples = 0
            train_loss = 0

            # ---------------------train---------------------
            for iter, (eeg, label) in enumerate(tqdm.tqdm(train_loader, position=0, leave=True), start=1):
                running_loss = 0.0
                # get the input
                eeg = eeg.to(cfg.device)
                label = label.to(cfg.device)

                pred = model_2D(eeg)
                loss = criterion(pred, label)
                train_loss += loss

                # backward
                optimizer_2D.zero_grad()  # clear the grad
                loss.backward()

                # gradient descent or adam step
                optimizer_2D.step()




            # ---------------------valid---------------------
            num_correct = 0
            num_samples = 0
            valid_loss = 0.0
            model_2D.eval()
            for iter, (eeg, label) in enumerate(tqdm.tqdm(valid_loader, position=0, leave=True), start=1):
                with torch.no_grad():
                    eeg = eeg.to(cfg.device)
                    label = label.to(cfg.device)
                    pred = model_2D(eeg)
                    loss = criterion(pred, label)
                    valid_loss = loss + valid_loss
                    _, predictions = pred.max(1)
                    num_correct += (predictions == label).sum()
                    num_samples += predictions.size(0)

            decoder_answer = float(num_correct) / float(num_samples) * 100

            print(f"kfold: {fold} epoch: {epoch},\n"
                    f"valid loss: {valid_loss / iter} , valid_decoder_answer: {decoder_answer}%\n")

            # Please note that for the densenet model_2D,
            # the result presented here is a classification accuracy of 1/128s rather than 1s
            if valid_loss_min>valid_loss / iter:
                valid_loss_min = valid_loss / iter
                savedir = './model_2D/'
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                saveckpt = savedir + '/fold' + str(fold) + '.ckpt'
                torch.save(model_2D.state_dict(), saveckpt)



    # I3D
    # get the dataset
    # inflate the model_2D

    valid_loss_min = 100

    model_3D = DenseNet_3D().to(cfg.device)

    if cfg.model_name == 'DenseNet_37-I3D':
        model_2D = DenseNet_2D().to(cfg.device)
        model_2D.load_state_dict(torch.load(saveckpt))
        weights_2D = model_2D.state_dict()
        weights_3D = model_3D.state_dict()

        for name, param in weights_2D.items():
            shape_2D = param.shape
            shape_3D = weights_3D[name].shape
            if shape_2D == shape_3D:
                weights_3D[name] = param
            else:
                param_3D = torch.zeros(shape_3D)
                time_shape = shape_3D[-3]
                for i in range(time_shape):
                    param_3D[:,:,i, :, :] = param / time_shape
                weights_3D[name] = param_3D

        model_3D.load_state_dict(weights_3D)


    train_dataset = AADdataset_1second(x_train, y_train)
    valid_dataset = AADdataset_1second(x_valid, y_valid)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.batch_size, shuffle=True)

    # set the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_3D = torch.optim.AdamW(model_3D.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---------------------train and valid-----------

    for epoch in range(cfg.epoch_num*3):

        # train the model_3D
        num_correct = 0
        num_samples = 0
        train_loss = 0

        # ---------------------train---------------------
        for iter, (eeg, label) in enumerate(tqdm.tqdm(train_loader, position=0, leave=True), start=1):
            running_loss = 0.0
            # get the input
            eeg = eeg.to(cfg.device)
            label = label.to(cfg.device)

            pred = model_3D(eeg)
            loss = criterion(pred, label)
            train_loss += loss

            # backward
            optimizer_3D.zero_grad()  # clear the grad
            loss.backward()

            # gradient descent or adam step
            optimizer_3D.step()

            _, predictions = pred.max(1)
            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)


        # ---------------------valid---------------------
        num_correct = 0
        num_samples = 0
        valid_loss = 0.0
        model_3D.eval()
        for iter, (eeg, label) in enumerate(tqdm.tqdm(valid_loader, position=0, leave=True), start=1):
            with torch.no_grad():
                eeg = eeg.to(cfg.device)
                label = label.to(cfg.device)
                pred = model_3D(eeg)
                loss = criterion(pred, label)
                valid_loss = loss + valid_loss
                _, predictions = pred.max(1)
                num_correct += (predictions == label).sum()
                num_samples += predictions.size(0)

        decoder_answer = float(num_correct) / float(num_samples) * 100

        print(f"kfold: {fold} epoch: {epoch},\n"
              f"valid loss: {valid_loss / iter} , valid_decoder_answer: {decoder_answer}%\n")


        if valid_loss_min > valid_loss / iter:
            valid_loss_min = valid_loss / iter
            savedir = './model_3D/'
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            saveckpt = savedir + '/fold' + str(fold) + '.ckpt'
            torch.save(model_3D.state_dict(), saveckpt)





def test_model(eegdata, eeglabel, fold):

# ----------------------initial model_3D------------------------

    # loading model_3D
    model_3D = DenseNet_3D().to(cfg.device)

    # test using the current folded data
    x_test, y_test = eegdata, eeglabel


    test_dataset = AADdataset_1second(x_test, y_test)
    # test the data one by one
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)


# -------------------------test--------------------------------------------
    # after some epochs, test model_3D
    savedir = './model_3D/'
    saveckpt = savedir + '/fold' + str(fold) + '.ckpt'
    test_acc = 0
    model_3D.load_state_dict(torch.load(saveckpt))
    model_3D.eval()
    total_num = 0
    for iter, (eeg, label) in enumerate(tqdm.tqdm(test_loader, position=0, leave=True), start=1):
        with torch.no_grad():

            # the between densenet and other models
            #
            eeg = eeg.to(cfg.device)
            label = label.to(cfg.device)
            pred = model_3D(eeg)

            _, predictions = pred.max(1)

            if predictions == label:
                test_acc += 1
            total_num = total_num + 1

    res = 100 * test_acc / total_num
    print('Fold %d test accuracy: %.3f %%' % (fold, res))


    return res