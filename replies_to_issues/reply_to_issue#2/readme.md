Thank you very much for the valuable feedback provided by charlespan110. Today, I reran the code and did not find the "close to 100%" result you mentioned. The output of the console can be found in "console_output.txt", and 'result.csv' presents all the results.

We doubt that you may have used the KUL dataset v0 instead of v1? This is important. Please verify the dataset you downloaded again. Due to the global algorithm used to remove eye artifacts, it has been proven that the results on the KUL dataset v0 may be significantly biased.

However, we did find that increasing the number of epochs can further improve the decoding accuracy (88.4% vs 84.8%). This is very helpful. When doing this project, we only used 30 epochs for all models except for the densenet-2d model. The densenet-2d model only used 10 epochs because running one epoch takes too long. We think this is fair because all models have seen the same times of dataset, and only the model we proposed has seen less data.

It is worth mentioning that even after training for 200 epochs, the results of the baseline model are still far worse than our proposed model training for 30 epoches.

Thank you again.