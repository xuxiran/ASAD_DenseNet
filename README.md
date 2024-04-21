# ASAD_DenseNet
------------------------20240321--------------------------------

This paper has been accepted by ICASSP 2024 and this is the website: https://ieeexplore.ieee.org/document/10448013

![1713698136124](https://github.com/xuxiran/ASAD_DenseNet/assets/48015859/46116aa2-7f2b-4b81-a94c-76c670b1cdb7)


------------------------20231123--------------------------------

We have attempted to use an 8th order Butterworth filter, which is an IIR filter, to replace 512th order FIR filter and obtained similar results (94.3% in a 1-second decision window). The implemented code has been synchronously updated to '3_processed'.

------------------------20230915--------------------------------

This project provides the implementation of the ASAD_DenseNet, with code to reproduce all the results reported in the paper: 
https://arxiv.org/abs/2309.07690

![image](https://github.com/xuxiran/ASAD_DenseNet/assets/48015859/5dd21b14-00a0-4194-9a57-297f39b04f37)


The '1_download_data' folder includes only the 'download_data.py' file, which is used to download the raw data. All reported results in the paper were obtained using the KUL dataset, which can be downloaded at https://zenodo.org/record/3377911. To simplify the data download process, the 'download_data.py' script is provided to help users quickly download the dataset. All data will be downloaded to the '../2_data' location, which is required for subsequent preprocessing and analysis.

The '2_data' folder contains all the downloaded data. It is currently empty to reduce file size, but will be populated with 16 'mat' files named 'S1.mat' to 'S16.mat' after running the 'download_data.py' script.

The '3_preprocess' folder includes all preprocessing files. 'EEG_2D.xlsx' contains the spatial arrangement of EEG electrodes, while 'EEG_map.mat' saves the labels of electrodes 1-64. 'preprocess.m' is the preprocessing execution file, which requires MATLAB to run and the eeglab toolbox can be downloaded from https://sccn.ucsd.edu/eeglab/download.php.

The '4_processed_data' folder includes all generated data. It is currently empty to reduce file size, but will be populated with two files, 'KUL_1D.mat' and 'KUL_2D.mat', after running the 'preprocess_IIR.m' script. They respectively correspond to the one-dimensional electrode arrangement EEG data and the EEG data that contains spatial information.

The '5_train_and_test_model' folder contains multiple model folders. It is divided into two folders corresponding to sections 4.1 and 4.2 of the paper. Running the code in each folder will obtain all the results in sections 4.1 and 4.2.

In each file, all data is divided into five folds and cross-validated in 'main.py'. This ensures that there are no testing set leakage issues. Specific training, validation, and testing are performed in 'train_valid_and_test.py', which calls various models and uses the AADdataset.py to establish the dataset.

The '6_statistic_analysis' folder includes all the plotted images. It is divided into two folders corresponding to sections 4.1 and 4.2 of the paper. Running the code in each folder will obtain all the statistical test results in sections 4.1 and 4.2 and enable plotting using MATLAB code. Please note that some results require manual counting of the outputs of each model, averaging over the five cross-validations, and filling in the average values in 'plot_3_a.m' and 'plot_3_b.m'. Here, we provide our calculated results directly. 'plot_3_a.m' and 'plot_3_b' can be used for statistical testing and plotting of the results in Figure 3 of the paper.

In summary, the codes for all results in the paper are presented in this project. If any difficulties are encountered during reproducing our work, please contact the email address 2001111407@stu.pku.edu.cn as soon as possible, and we will respond promptly. Thank you for your time.

