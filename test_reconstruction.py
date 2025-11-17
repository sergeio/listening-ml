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

datafile, sr = librosa.load('ah.au', sr=None)
spec = librosa.stft(datafile, n_fft=2048, hop_length=2048 // 2)
mag, phase = librosa.magphase(spec)

data = []
for t in range(len(mag[0])):
    t_slice = [x[t] for x in mag]
    data.append(t_slice[10:600])

xs = torch.Tensor(np.array(data[:-1]))
ys = torch.Tensor(np.array(data[1:]))

t_size = len(data)-1
f_size = len(data[0])

W = torch.randn(f_size, f_size, requires_grad=True)
W.data *= .001
eps = 1e-32


epochs = 100000
model_file = f'W{epochs}.pkl-{this_file}'
try:
    assert False
    # Uncomment if you want to enable using saved Ws from pkl files
    # W = pickle.load(open(model_file, 'rb'))
    # print(f'read model {model_file}')
except:
    for j in range(epochs):
        rate = .03
        loss = 0
        logits = xs @ W
        loss = ((logits - ys)**2).mean()

        # Backward pass
        W.grad = None
        loss.backward()
        W.data -= rate * W.grad

        if not j % (epochs // 100):
            print(f'{j} {loss.data=}')

    pickle.dump(W, open(model_file, 'wb'))
    print(f'wrote model {model_file}')

# In case you want to cheat and predict os using all xs (instead of starting
# with xs[0] and generating 1 frame at a time:
# os = xs @ W.detach()
# os = torch.cat((xs[0].unsqueeze(0), os), dim=0)

os = [xs[0]]
for i in range(1, t_size):
  os.append(os[i-1] @ W.detach())

out_spec = []
for f in range(f_size):
    f_slice = [0] * 10
    for t in range(t_size):
        # Arguably, we're still cheating by using the original phase
        f_slice.append(complex(os[t][f],0) * phase[f][t])
    f_slice += [0] * (1024 - 600)
    out_spec.append(f_slice)

f_reconstructed = librosa.istft(np.array(out_spec), n_fft=2048, hop_length=2048 // 2)
soundfile.write('test.wav', f_reconstructed, sr)

