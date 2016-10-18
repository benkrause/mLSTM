This code provides an implementation of multiplicative LSTM and a stacked LSTM baseline which can be quickly set up for character level language modelling of the Hutter prize dataset. Slight modifcations would allow application to other character level modelling tasks.

Instructions for use 

1. Download file enwik8.zip from http://www.mattmahoney.net/dc/textdata , and include unzipped file with name "enwik8" in directory. 

2. Run mLSTMhutter.m or LSTMhutter.m from MATLAB to train mLSTM or LSTM model on data

3. Run mLSTMeval.m or LSTMeval.m from MATLAB to evaluate model trained in previous step on test set

4. Run mLSTMdynamic.m or LSTMdynamic.m from MATLAB to dynamically evaluate model on test set

Files

mLSTMhutter.m 
-Trains a multiplicative LSTM on the hutter prize dataset, saves network parameter values and writes continually to a log file. Takes ~2 days to run on GTX 970 GPU.

mLSTMeval.m 
-Performs static evaluation of mLSTM on the test set, loading the network parameters saved during training. Takes a few minutes to run on GTX 970 GPU.

mLSTMdynamic.m 
-Performs dynamic evaluation of mLSTM on the test set, loading the network parameters saved during training. Takes ~1 day to run on GTX 970 GPU.


LSTMhutter.m 
-Trains an LSTM on the hutter prize dataset, saves parameter values and writes continually to a log file. Takes ~2 days to run on GTX 970 GPU.

LSTMeval.m 
-Performs static evaluation of LSTM on the test set, loading the network parameters saved during training. Takes a few minutes to run on GTX 970 GPU.

LSTMdynamic.m 
-Performs dynamic evaluation of LSTM on the test set, loading the network parameters saved during training. Takes ~1 day to run on GTX 970 GPU.

processtextfile.m
-reads training set text file into appropriate format


Dependencies: 
MATLAB 2014a or newer, with parellel computing toolbox
CUDA enabled GPU with atleast 4GB of RAM

If no GPU is available, code can be run on CPU by commenting out "gpuDevice(1)" command at the top of the files, and removing comments from "gpuArray" and "gather" functions at the end of the files. However, running the experiments on a CPU using the full training set would be quite slow. Training set size can be reduced by setting the "maxtrain" variable.
