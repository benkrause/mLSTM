


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

        self.l1.b.data[2::4] = 3
        Wembd = np.random.uniform(-.2, .2, self.embed.W.data.shape)
        Wembd =Wembd.astype(dtype=np.float32)
        self.embed.W.data = Wembd

        self.WhxWN.W.data = ortho_init(self.WhxWN.W.data.shape)
        norm = np.linalg.norm(self.WhxWN.W.data, axis=1)
        self.WhxWN.g.data = norm

        self.WmxWN.W.data = ortho_init(self.WmxWN.W.data.shape)
        norm = np.linalg.norm(self.WmxWN.W.data, axis=1)
        self.WmxWN.g.data = norm

        self.WmhWN.W.data = ortho_init(self.WmhWN.W.data.shape)
        norm = np.linalg.norm(self.WmhWN.W.data, axis=1)
        self.WmhWN.g.data = norm

        self.WhmWN.W.data = ortho_init(self.WhmWN.W.data.shape)
        norm = np.linalg.norm(self.WhmWN.W.data, axis=1)
        self.WhmWN.g.data = norm

        self.l2.W.data= ortho_init(self.l2.W.data.shape)

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
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=200,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=40,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')

    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')

    parser.add_argument('--file', default="enwik8",
                        help='path to text file for training')
    parser.add_argument('--unit', '-u', type=int, default=2800,
                        help='Number of LSTM units')
    parser.add_argument('--embd', type=int, default=400,
                        help='Number of embedding units')
    parser.add_argument('--hdrop', type=float, default=0.2,
                        help='hidden state dropout (variational)')
    parser.add_argument('--edrop', type=float, default=0.5,
                        help='embedding dropout')

    args = parser.parse_args()

    nembd = args.embd
    #number of training iterations per model save, log write, and validation set evaluation
    interval =100

    pdrop = args.hdrop

    pdrope = args.edrop

    #initial learning rate
    alpha0 = .001
    #inverse of linear decay rate towards 0
    dec_it = 12*9000
    #minimum learning rate
    alpha_min = .00007

    #first ntrain words of dataset will be used for training
    ntrain = 90000000


    seqlen = args.bproplen
    nbatch = args.batchsize

    filename= args.file

    text,mapping = get_char(filename)
    sequence = np.array(text).astype(np.int32)

    itrain =sequence[0:ntrain]
    ttrain = sequence[1:ntrain+1]
    fullseql=int(ntrain/nbatch)

    itrain = itrain.reshape(nbatch,fullseql)
    ttrain = ttrain.reshape(nbatch,fullseql)

    #doesn't use full validations set
    nval = 500000
    ival = sequence[ntrain:ntrain+nval]
    tval = sequence[ntrain+1:ntrain+nval+1]

    ival = ival.reshape(ival.shape[0]//1000,1000)
    tval = tval.reshape(tval.shape[0]//1000,1000)
    #test = sequence[ntrain+nval:ntrain+nval+ntest]


    nvocab = max(sequence) + 1  # train is just an array of integers
    print('#vocab =', nvocab)

    # Prepare an RNNLM model
    rnn = RNNForLM(nvocab, args.unit,args.embd)
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # Set up an optimizer
    optimizer = Adam(alpha=alpha0)
    optimizer.setup(model)
    resultdir = args.out

    print('starting')
    nepoch = args.epoch

    start = 0
    loss_sum = 0;

    if not os.path.isdir(resultdir):
        os.mkdir(resultdir)

    vloss = test(rnn,ival,tval)
    vloss= (1.4427*vloss)
    f = open(os.path.join(resultdir,'log'), 'w')
    outstring = "Initial Validation loss (bits/word): " + str(vloss) + '\n'
    f.write(outstring)
    f.close()

    i=0
    epoch_num = 0
    it_num = 0

    while True:
        # Get the result of the forward pass.
        fin = start+seqlen

        if fin>(itrain.shape[1]):
            start = 0
            fin = start+seqlen
            epoch_num = epoch_num+1
            if epoch_num== nepoch:
                break

        inputs = itrain[:,start:fin]
        targets = ttrain[:,start:fin]
        start = fin

        inputs = Variable(inputs)
        targets = Variable(targets)

        targets.to_gpu()
        inputs.to_gpu()
        it_num+=1
        loss = 0
        rnn.applyWN()

        #make hidden dropout mask
        mask = cp.zeros((inputs.shape[0],args.unit),dtype = cp.float32)
        ind = cp.nonzero(cp.random.rand(inputs.shape[0],args.unit)>pdrop)
        mask[ind] = 1/(1-pdrop)

        #make embedding dropout mask
        mask2 = cp.zeros((inputs.shape[0],nembd),dtype = cp.float32)
        ind = cp.nonzero(cp.random.rand(inputs.shape[0],nembd)>pdrope)
        mask2[ind] = 1/(1-pdrope)

        for j in range(seqlen):

            output = rnn(inputs[:,j],mask,mask2)
            loss = loss+ F.softmax_cross_entropy(output,targets[:,j])

        loss = loss/(seqlen)

        # Zero all gradients before updating them.
        rnn.zerograds()
        loss_sum += loss.data

        # Calculate and update all gradients.
        loss.backward()
        s = 0;

        # Use the optmizer to move all parameters of the network
        # to values which will reduce the loss.
        optimizer.update()
        #decays learning rate linearly
        optimizer.alpha = alpha0*(dec_it-it_num)/float(dec_it)
        #prevents learning rate from going below minumum
        if optimizer.alpha<alpha_min:
            optimizer.alpha = alpha_min

        loss.unchain_backward()

        if ((i+1)%interval) ==0:
            rnn.reset_state()
            vloss = test(rnn,ival,tval)

            #converts to binary entropy
            vloss= (1.4427*vloss)
            loss_sum = (1.4427*loss_sum/interval)

            serializers.save_npz(os.path.join(resultdir,'model'),rnn)

            outstring = "Training iteration: " + str(i+1) + " Training loss (bits/char): " + str(loss_sum) + " Validation loss (bits/word): " + str(vloss) + '\n'
            f = open(os.path.join(resultdir,'log'), 'a')
            f.write(outstring)
            f.close()
            print("Training iteration: " + str(i+1))
            print('training loss: ' + str(loss_sum))
            print('validation loss: ' + str(vloss))
            loss_sum=0

        i+=1

main()
