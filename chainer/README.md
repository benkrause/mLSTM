This is a chainer implementation of weight normalized multiplicative LSTM with variational dropout.

requirements: python3, chainer==1.24

does not run in more recent versions of chainer, install chainer with:

pip install chainer==1.24

this code is by default set to run with text8 and Hutter Prize. 

Hutter Prize can be downloaded here: http://mattmahoney.net/dc/enwik8.zip
text8 can be downloaded here: http://mattmahoney.net/dc/text8.zip

To train the model, put the uzipped text file (either text8 or enwik8) in the directory and use:

python train.py --file filename

The default settings were used to obtain 1.24 bits/char on Hutter Prize and 1.27 bits/char on text8. We did not save the initialization seed, but it should hopefully be possible to reproduce similar results. The model takes about 1 week to train on a GTX 1080 TI, and requires about 9GB of GPU memory. The model and log file are stored in the directory specified by --out ("result" by default). The model is saved intermittently throughout training at every log update. To evaluate the test set error after training, run:

python eval.py --file filename

train.py assumes the training set is the first 90M characters, and eval.py assumes the test set is the last 5M characters. This is the case for both Hutter prize and text8. 

To train faster and with less memory you could try:

python train.py --file filename --epoch 10 --edrop 0.2 --unit 1900 --bproplen 100

This configuration should finish roughly 5 times faster and use under 4GB of GPU memory, but will not obtain as strong results. The hidden and embedding sizes from training must be specified during evaluation, so if you use the above training configuration, evaluate with:

python eval.py --file filename --unit 1900















