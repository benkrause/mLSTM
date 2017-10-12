import argparse
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import training
import sys
import cupy as cp
import os
from mLSTMWN import mLSTM
from WN import WN
from chainer.optimizers import Adam
from chainer import serializers

def get_char(fname):
    fid = open(fname,'rb')
    byte_array = fid.read()
    text = [0]*len(byte_array)
    for i in range(0,len(byte_array)):
        text[i] = int(byte_array[i])
    unique = list(set(text))
    unique.sort()

    mapping = dict(zip(unique,list(range(0,len(unique)))))
    for i in range(0,len(text)):
        text[i] = mapping[text[i]]
    return text, mapping

def ortho_init(shape):
    # From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120

    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    a=a.astype(dtype=np.float32)

    u, _, v = np.linalg.svd(a, full_matrices=False)

    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q

class RNNForLM(chainer.Chain):

    def __init__(self, nvocab, nunits, train=True):
        super(RNNForLM, self).__init__(
            embed=L.EmbedID(nvocab, 400),
            WhxWN = WN(400,nunits*4),
            WmxWN = WN(400,nunits),
            WmhWN = WN(nunits,nunits),
            WhmWN = WN(nunits,nunits*4),

            l1=mLSTM(out_size=nunits),
            l2=L.Linear(nunits, nvocab)
        )
        nparam = 0
        for param in self.params():
            print(param.data.shape)
            nparam+=param.data.size
        nparam+=param.data.size
        print('nparam')
        print(nparam)

        self.train = train

    def reset_state(self):
        self.l1.reset_state()

    def applyWN(self):
        self.Whx = self.WhxWN()
        self.Wmx = self.WmxWN()
        self.Wmh = self.WmhWN()
        self.Whm = self.WhmWN()

    def __call__(self, x,mask,mask2):


        h0 = self.embed(x)*mask2

        h1 = self.l1(h0,self.Whx,self.Wmx,self.Wmh,self.Whm)
        h1=h1*mask
        self.l1.h = h1
        y = self.l2(h1)

        return y

def test(model,inputs,targets):
    inputs = Variable(inputs)
    targets = Variable(targets)

    targets.to_gpu()
    inputs.to_gpu()
    model.applyWN()
    model.train = False
    loss=0
    for j in range(inputs.shape[1]):
        output = model(inputs[:,j],1,1)
        loss = loss+ F.softmax_cross_entropy(output,targets[:,j])
        loss.unchain_backward()

    model.train=True

    model.reset_state()

    finalloss = loss.data/inputs.shape[1]
    return finalloss

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='test or val batch size, 5M mod batchsize must be 0')

    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--model', '-o', default='result',
                        help='Directory to load model from')

    parser.add_argument('--file', default="enwik8",
                        help='path to text file for testing')
    parser.add_argument('--unit', '-u', type=int, default=2800,
                        help='Number of LSTM units, must match model')
    parser.add_argument('--embd', type=int, default=400,
                        help='Number of embedding units, must match model')
    parser.add_argument('--val', action='store_true',
                        help='set for validation error, test by default')

    args = parser.parse_args()

    nembd = args.embd

    nbatch = args.batchsize

    filename= args.file

    text,mapping = get_char(filename)
    sequence = np.array(text).astype(np.int32)

    if args.val:
        start = 90000000-1
    else:
        start = 95000000-1
    neval = 5000000

    ival = sequence[start:start+neval]
    tval = sequence[start+1:start+neval+1]

    #uses subset of validation set
    ival = ival.reshape(args.batchsize,ival.shape[0]//args.batchsize)
    tval = tval.reshape(args.batchsize,tval.shape[0]//args.batchsize)
    #test = sequence[ntrain+nval:ntrain+nval+ntest]
    nvocab = max(sequence) + 1  # train is just an array of integers
    print('#vocab =', nvocab)
    # Prepare an RNNLM model
    rnn = RNNForLM(nvocab, args.unit,args.embd)
    modelname = os.path.join(args.model,'model')
    serializers.load_npz(modelname, rnn)
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    print('starting')

    start = 0
    loss_sum = 0;

    vloss = test(rnn,ival,tval)
    vloss= (1.4427*vloss)
    print('loss (bits/char): ' + str(vloss))

main()
