c3;
% add your eeglab address,or you can add the path to dir
% addpath(genpath('D:\eeglab_current\eeglab2022.0'));
% produce 2*2*2=8 data
data_types = {'1D','2D'};
paralen = 60*128;
sbnum = 16;
trnum = 8;

dataset = 'KUL';

data1D_name = [dataset '_1D.mat'];
data2D_name = [dataset '_2D.mat'];

EEG = zeros(sbnum,trnum,6*paralen,64);
ENV = zeros(sbnum,trnum,6*paralen,1);

rawdir=['../2_data'];

for sb = 1:sbnum
    load([rawdir filesep 'S' num2str(sb) '.mat']);

    for tr = 1:trnum
        disp(['preprocess_data      subject:' num2str(sb) '   trial:' num2str(tr)]);
        trial = trials{tr};%read the trialnum's trial

        tmp = double(trial.RawData.EegData);

        eegtrain = tmp(1:6*paralen,:)';


        fs = 128;
        EEG_trial = pop_importdata('dataformat','array','nbchan',0,'data','eegtrain','srate',fs,'pnts',0,'xmin',0);

        [EEG_trial,com,b] = pop_eegfiltnew(EEG_trial, 14,31,512,0,[],0);

        % verify the filter
        % [Pxx, F] = spectopo(EEG_trial.data, 0, 128, 'freqrange', [] );
        eegtrain = EEG_trial.data';

        % mean and std
        eegtrain = (eegtrain-mean(eegtrain,2))./std(eegtrain,0,2);

        % give label
        if trial.attended_ear=='L'
            labeltrain = ones(6*paralen,1);
        else
            labeltrain = zeros(6*paralen,1);
        end

        EEG(sb,tr,:,:) = eegtrain;
        ENV(sb,tr,:,:) = labeltrain;
    end

end

save(['../4_processed_data/' data1D_name],'EEG','ENV');


% expand 1d to 2d

load(['../4_processed_data/' data1D_name]);
eeglen = size(ENV,2);
EEG_2D = zeros(sbnum,trnum,6*paralen,10,11);


[~,map,~] = xlsread(['EEG_2D.xlsx']); % the channel position
load('EEG_map.mat') % the channel order
axis = zeros(64,2);
for cha = 1:64
    for w = 1:10
        for h = 1:11
            if strcmp(EEGMAP{cha},map{w,h})==1
                axis(cha,1) = w;
                axis(cha,2) = h;
            end
        end
    end
end

for cha = 1:64
    EEG_2D(:,:,:,axis(cha,1),axis(cha,2)) = EEG(:,:,:,cha);
end
EEG = EEG_2D;
save(['../4_processed_data/' data2D_name],'EEG','ENV');







