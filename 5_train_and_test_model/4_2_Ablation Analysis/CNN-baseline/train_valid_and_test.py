from AADdataset import AADdataset_1point,AADdataset_1second
from torch.utils.data import DataLoader
from model import CNN_baseline, CNN_2D, DenseNet_37
import tqdm
import torch
import config as cfg
import torch.nn as nn
from sklearn.model_selection import KFold,train_test_split
from torch.utils.tensorboard import SummaryWriter
import os

writer = SummaryWriter()


# train the model for every subject
def train_valid_model(eegdata, eeglabel, sb, fold):

# ----------------------initial model------------------------
    valid_loss_min = 100
    if cfg.model_name == 'CNN_baseline':
        # loading model
        model = CNN_baseline().to(cfg.device)
    elif cfg.model_name == 'CNN_2D':
        # loading model
        model = CNN_2D().to(cfg.device)
    elif cfg.model_name == 'DenseNet_37':
        model = DenseNet_37().to(cfg.device)

    # Train and test using the current folded data
    x_train_val, y_train_val = eegdata, eeglabel
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=2024)


    # get the dataset
    if cfg.model_name == 'CNN_baseline':
        train_dataset = AADdataset_1second(x_train, y_train)
        valid_dataset = AADdataset_1second(x_valid, y_valid)
    else:
        train_dataset = AADdataset_1point(x_train, y_train)
        valid_dataset = AADdataset_1point(x_valid, y_valid)


    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=cfg.batch_size, shuffle=True)


    # set the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


# ---------------------train and valid-----------

    for epoch in range(cfg.epoch_num):

        # train the model
        num_correct = 0
        num_samples = 0
        train_loss = 0

        # ---------------------train---------------------
        for iter, (eeg, label) in enumerate(tqdm.tqdm(train_loader, position=0, leave=True), start=1):
            running_loss = 0.0
            # get the input
            eeg = eeg.to(cfg.device)
            label = label.to(cfg.device)

            pred = model(eeg)
            loss = criterion(pred, label)
            train_loss += loss

            # backward
            optimizer.zero_grad()  # clear the grad
            loss.backward()

            # gradient descent or adam step
            optimizer.step()

            _, predictions = pred.max(1)
            num_correct += (predictions == label).sum()
            num_samples += predictions.size(0)

        decoder_answer = float(num_correct) / float(num_samples) * 100

        # Record the results of training
        trainloss = 'sb' + str(sb) + '/Train_Loss'
        train_decoder_answer = 'sb' + str(sb) + '/Train_decoder_answer'
        writer.add_scalar(trainloss, train_loss / iter, epoch // 1)
        writer.add_scalar(train_decoder_answer, decoder_answer, epoch // 1)


        # ---------------------valid---------------------
        num_correct = 0
        num_samples = 0
        valid_loss = 0.0
        model.eval()
        for iter, (eeg, label) in enumerate(tqdm.tqdm(valid_loader, position=0, leave=True), start=1):
            with torch.no_grad():
                eeg = eeg.to(cfg.device)
                label = label.to(cfg.device)
                pred = model(eeg)
                loss = criterion(pred, label)
                valid_loss = loss + valid_loss
                _, predictions = pred.max(1)
                num_correct += (predictions == label).sum()
                num_samples += predictions.size(0)

        decoder_answer = float(num_correct) / float(num_samples) * 100

        validloss = 'sb' + str(sb) + '/Valid_Loss'
        valid_decoder_answer = 'sb' + str(sb) + '/Valid_decoder_answer'
        writer.add_scalar(validloss, valid_loss / iter, epoch // 1)
        writer.add_scalar(valid_decoder_answer, decoder_answer, epoch // 1)
        print(f"sb: {sb}, kfold: {fold} epoch: {epoch},\n"
                f"valid loss: {valid_loss / iter} , valid_decoder_answer: {decoder_answer}%\n")

        # Please note that for the densenet model,
        # the result presented here is a classification accuracy of 1/128s rather than 1s
        if valid_loss_min>valid_loss / iter:
            valid_loss_min = valid_loss / iter
            savedir = './model_sb/sb' + str(sb)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            saveckpt = savedir + '/fold' + str(fold) + '.ckpt'
            torch.save(model.state_dict(), saveckpt)





def test_model(eegdata, eeglabel, sb, fold):

# ----------------------initial model------------------------

    if cfg.model_name == 'CNN_baseline':
        # loading model
        model = CNN_baseline().to(cfg.device)
    elif cfg.model_name == 'CNN_2D':
        # loading model
        model = CNN_2D().to(cfg.device)
    elif cfg.model_name == 'DenseNet_37':
        model = DenseNet_37().to(cfg.device)

    # test using the current folded data
    x_test, y_test = eegdata, eeglabel

    # tough the train and valid process exist difference
    # the test_data is same,one second by one second
    test_dataset = AADdataset_1second(x_test, y_test)
    # test the data one by one
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)


# -------------------------test--------------------------------------------
    # after some epochs, test model
    savedir = './model_sb/sb' + str(sb)
    saveckpt = savedir + '/fold' + str(fold) + '.ckpt'
    test_acc = 0
    model.load_state_dict(torch.load(saveckpt))
    model.eval()
    total_num = 0
    for iter, (eeg, label) in enumerate(tqdm.tqdm(test_loader, position=0, leave=True), start=1):
        with torch.no_grad():

            # the between densenet and other models
            #
            if cfg.model_name == 'densenet':
                eeg = eeg.squeeze(0)
                label = label.squeeze(0)

            eeg = eeg.to(cfg.device)
            label = label.to(cfg.device)
            pred = model(eeg)

            _, predictions = pred.max(1)
            if cfg.model_name == 'densenet':
                counts = torch.zeros(cfg.categorie_num)
                for i in range(cfg.categorie_num):
                    for j in range(predictions.size(0)):
                        if predictions[j] == i:
                            counts[i] += 1
                decode_ans = torch.argmax(counts)
            else:
                decode_ans = predictions

            if decode_ans == label:
                test_acc += 1
            total_num = total_num + 1

    res = 100 * test_acc / total_num
    print('Subject %d Fold %d test accuracy: %.3f %%' % (sb, fold, res))


    return res