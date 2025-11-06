import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile
import random
import torch
import torch.nn.functional as F
import os

this_file = os.path.basename(__file__)

datafile, sr = librosa.load('andrej.au', sr=None)
spec = librosa.stft(datafile, n_fft=2048, hop_length=2048 // 2)
mag, phase = librosa.magphase(spec)
BOUND_LOW = 20
BOUND_HIGH = 400

data = []
for t in range(len(mag[0])):
    t_slice = [x[t] for x in mag]
    data.append(t_slice[BOUND_LOW:BOUND_HIGH])

CONTEXT_WINDOW = 4
#xs = torch.Tensor(np.array(data[:-1]))
#xs = torch.Tensor(np.array([sum(list(ntuple),[]) for ntuple in zip(data[:-CONTEXT_WINDOW], data[1:-3], data[2:-2], data[3:-1])])) #change this line when changing window size
xs = torch.Tensor(np.array([sum(list(ntuple),[]) for ntuple in zip(*[data[i:i-CONTEXT_WINDOW] for i in range(CONTEXT_WINDOW)])]))
ys = torch.Tensor(np.array(data[CONTEXT_WINDOW:]))
print(f'{xs.shape=}, {ys.shape=}')

t_size = len(data)-CONTEXT_WINDOW
f_size = len(data[0])

W1 = torch.randn(f_size*CONTEXT_WINDOW, f_size, requires_grad=True)
b1 = torch.randn(f_size, requires_grad=True)
# W2 = torch.randn(f_size*2, f_size, requires_grad=True)
# b2 = torch.randn(f_size, requires_grad=True)
eps = 1e-32

epochs = 60000
model_file = f'W{epochs}.pkl-{this_file}'
try:
    # assert False
    state = pickle.load(open(model_file, 'rb'))
    print(f'read model {model_file}')
    #(W1, b1, W2, b2) = state
    (W1, b1) = state
except:
    for j in range(epochs):
        rate = 0.3
        # Forward pass
        logits = xs @ W1 + b1
        # logits = (xs @ W1 + b1)
        # logits = logits.tanh()
        # logits = logits @ W2 + b2
        loss = ((logits - ys)**2).mean() + 0.1*(logits**2).mean()

        # Backward pass
        W1.grad = None
        b1.grad = None
        # W2.grad = None
        # b2.grad = None
        loss.backward()
        W1.data -= rate * W1.grad
        b1.data -= rate * b1.grad
        # W2.data -= rate * W2.grad
        # b2.data -= rate * b2.grad

        if not j % (epochs // 100):
            print(f'{j} {loss.data=}')

    #state = (W1, b1, W2, b2)
    state = (W1, b1)
    pickle.dump(state, open(model_file, 'wb'))
    print(f'wrote model {model_file}')

W1 = W1.detach()
b1 = b1.detach()
# W2 = W2.detach()
# b2 = b2.detach()

num_freq = BOUND_HIGH-BOUND_LOW

#os = (xs @ W1 + b1).tanh() @ W2 + b2
os = xs @ W1 + b1
print(xs[0].shape)
#os = torch.cat((xs[0][:num_freq].unsqueeze(0), xs[0][num_freq:2*num_freq].unsqueeze(0), xs[0][2*num_freq:3*num_freq].unsqueeze(0), xs[0][3*num_freq:].unsqueeze(0), os), dim=0) #CHANGE WHEN CHANGING CONTEXT_WINDOW
os = torch.cat((*[xs[0][i*num_freq:(i+1)*num_freq].unsqueeze(0) for i in range(CONTEXT_WINDOW)], os), dim=0)


#os = [xs[0][:num_freq], xs[0][num_freq:num_freq*2], xs[0][num_freq*2:num_freq*3], xs[0][num_freq*3:]] #change this line when changing window size
# os = [xs[0][i*num_freq:(i+1)*num_freq] for i in range(CONTEXT_WINDOW)]

# for i in range(CONTEXT_WINDOW, t_size):
#     #os.append((os[i-1] @ W1 + B1).tanh() @ W2 + B2)
#     # input_os = torch.cat((os[i-CONTEXT_WINDOW], os[i-3], os[i-2], os[i-1]), dim=0) #change this line when changing window size
#     input_os = torch.cat((*[os[i+j-CONTEXT_WINDOW] for j in range(CONTEXT_WINDOW)],), dim=0)
#     temp = (input_os) @ W1
#     os.append(temp + b1)


for t in range(20):
    plt.figure(figsize=(12,6))
    plt.plot(os[t], c='r')
    plt.plot(ys[t], c='g', alpha=.6)
    plt.show()

out_spec = []
for f in range(f_size):
    f_slice = [0] * BOUND_LOW
    for t in range(t_size):
        f_slice.append(complex(os[t][f],0) * phase[f][t])
    f_slice += [0] * (1024 - BOUND_HIGH)
    out_spec.append(f_slice)

f_reconstructed = librosa.istft(np.array(out_spec), n_fft=2048, hop_length=2048 // 2)
soundfile.write('test.wav', f_reconstructed, sr)

