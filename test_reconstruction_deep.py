import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile
import random
import torch
from torch import nn
import os

this_file = os.path.basename(__file__)

datafile, sr = librosa.load('wawu.au', sr=None)
spec = librosa.stft(datafile, n_fft=2048, hop_length=2048 // 2)
mag, phase = librosa.magphase(spec)

data = []
for t in range(len(mag[0])):
    t_slice = [x[t] for x in mag]
    data.append(t_slice)

g = torch.Generator().manual_seed(1)

CONTEXT_WINDOW = 1

#xs = torch.Tensor(np.array(data[:-1])) # assumes context window of 1
xs = torch.Tensor(np.array([sum(list(ntuple),[]) for ntuple in zip(*[data[i:i-CONTEXT_WINDOW] for i in range(CONTEXT_WINDOW)])]))
ys = torch.Tensor(np.array(data[CONTEXT_WINDOW:]))
print(f'{xs.shape=}, {ys.shape=}')

t_size = len(data)-CONTEXT_WINDOW
f_size = len(data[0])

layers = [nn.Linear(f_size*CONTEXT_WINDOW, f_size, bias=False), nn.BatchNorm1d(f_size)]
parameters = [p for layer in layers for p in layer.parameters()]
eps = 1e-32

os = []

epochs = 10000
# model_file = f'W{epochs}.pkl-{this_file}'
try:
    assert False
    # state = pickle.load(open(model_file, 'rb'))
    # print(f'read model {model_file}')
    # (W1, b1) = state
except:
    for j in range(epochs):
        os = xs
        for layer in layers:
            os = layer(os)
        loss = ((os - ys)**2).mean()

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        lr = 0.04
        for p in parameters:
            p.data -= lr * p.grad

        if not j % (epochs // 10):
            print(f'{j} {loss.data=}')

    # state = (W1, b1)
    # pickle.dump(state, open(model_file, 'wb'))
    # print(f'wrote model {model_file}')



# os = torch.cat((xs[0][:num_freq].unsqueeze(0), xs[0][num_freq:2*num_freq].unsqueeze(0), xs[0][2*num_freq:3*num_freq].unsqueeze(0), xs[0][3*num_freq:].unsqueeze(0), os), dim=0) #CHANGE WHEN CHANGING CONTEXT_WINDOW
os = torch.cat((*[xs[0][i*num_freq:(i+1)*num_freq].unsqueeze(0) for i in range(CONTEXT_WINDOW)], os), dim=0)


## os = [xs[0][:num_freq], xs[0][num_freq:num_freq*2], xs[0][num_freq*2:num_freq*3], xs[0][num_freq*3:]] #change this line when changing window size
# os = [xs[0][i*num_freq:(i+1)*num_freq] for i in range(CONTEXT_WINDOW)]

# generate output using previous output
for i in range(CONTEXT_WINDOW, t_size):
    input_os = torch.cat((*[os[i+j-CONTEXT_WINDOW] for j in range(CONTEXT_WINDOW)],), dim=0)
    for layer in layers:
        input_os = layer(input_os)
    os.append(input_os)


out_spec = []
for f in range(f_size):
    f_slice = []
    for t in range(t_size):
        f_slice.append(complex(os[t][f],0) * phase[f][t])
        #t_slice.append(os[t][f])
    out_spec.append(f_slice)

f_reconstructed = librosa.istft(np.array(out_spec), n_fft=2048, hop_length=2048 // 2)
soundfile.write('test.wav', f_reconstructed, sr)

