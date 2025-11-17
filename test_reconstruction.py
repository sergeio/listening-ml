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

datafile, sr = librosa.load('ma.au', sr=None)
spec = librosa.stft(datafile, n_fft=2048, hop_length=2048 // 2)
mag, phase = librosa.magphase(spec)

data = []
for t in range(len(mag[0])):
    t_slice = [x[t] for x in mag]
    data.append(t_slice[10:600])

# for j, o in enumerate(data):
#     data[j] = [freq if i < 600 and i > 10 else 0 for i, freq in enumerate(o)]

xs = torch.Tensor(np.array(data[:-1]))
ys = torch.Tensor(np.array(data[1:]))

t_size = len(data)-1
f_size = len(data[0])

W = torch.randn(f_size, f_size, requires_grad=True)
W.data *= .001
eps = 1e-32


epochs = 10000
model_file = f'W{epochs}.pkl-{this_file}'
try:
    assert False
    # W = pickle.load(open(model_file, 'rb'))
    # print(f'read model {model_file}')
except:
    for j in range(epochs):
        rate = .04
        loss = 0
        # Forward pass
        # loss = F.cross_entropy(xs @ W, ys)
        logits = xs @ W
        loss = ((logits - ys)**2).mean()

        #logits = logits - logits.max()
        #counts = logits.exp() # + eps

        # Backward pass
        W.grad = None
        loss.backward()
        W.data -= rate * W.grad

        if not j % (epochs // 100):
            print(f'{j} {loss.data=}')

    pickle.dump(W, open(model_file, 'wb'))
    print(f'wrote model {model_file}')

# os = xs @ W.detach()
# os = torch.cat((xs[0].unsqueeze(0), os), dim=0)

os = [xs[0]]
for i in range(1, t_size):
  os.append(os[i-1] @ W.detach())

# os = xs

# os = data
# for j, o in enumerate(os):
#     os[j] = [freq if i < 600 and i > 10 else 0 for i, freq in enumerate(o)]

# for t in range(t_size):
#     plt.figure(figsize=(20,12))
#     plt.plot(os[t], c='r')
#     plt.plot(ys[t], c='g', alpha=.6)
#     plt.show()

out_spec = []
for f in range(f_size):
    f_slice = [0] * 10
    for t in range(t_size):
        f_slice.append(complex(os[t][f],0) * phase[f][t])
        # f_slice.append(complex(os[t][f],0) * 1)
        #t_slice.append(os[t][f])
    f_slice += [0] * (1024 - 600)
    out_spec.append(f_slice)

f_reconstructed = librosa.istft(np.array(out_spec), n_fft=2048, hop_length=2048 // 2)
soundfile.write('test.wav', f_reconstructed, sr)

