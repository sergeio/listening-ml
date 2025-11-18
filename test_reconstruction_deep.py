import numpy as np
import librosa
import soundfile
import torch
from torch import nn

datafile, sr = librosa.load('fraug.au', sr=None)
spec = librosa.stft(datafile, n_fft=2048, hop_length=2048 // 2)
freq_mag, freq_phase = librosa.magphase(spec)

f_size = len(freq_mag)
t_size = len(freq_mag[0])

data = []
for t in range(t_size):
    t_slice = [f[t] for f in freq_mag]
    data.append(t_slice)

CONTEXT_WINDOW = 5

xs = torch.Tensor(data[:-1])
ys = torch.Tensor(data[CONTEXT_WINDOW:])

HIDDEN_SIZE = 500
layers = [
    nn.Linear(f_size * CONTEXT_WINDOW, HIDDEN_SIZE, bias=False),
    nn.BatchNorm1d(HIDDEN_SIZE),
    nn.Tanh(),

    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False),
    nn.BatchNorm1d(HIDDEN_SIZE),
    nn.Tanh(),

    nn.Linear(HIDDEN_SIZE, f_size, bias=False),
    nn.BatchNorm1d(f_size),
]
parameters = [p for layer in layers for p in layer.parameters()]

def overlapping_windows(array, window_size):
    num_windows = len(array) - window_size + 1
    # i is the start of each window
    return [tuple(array[i:i+window_size]) for i in range(num_windows)]
assert overlapping_windows(list(range(5)), 3) == [(0, 1, 2), (1, 2, 3), (2, 3, 4)]

epochs = 20000
for j in range(epochs):
    os = [torch.cat(w) for w in overlapping_windows(xs, CONTEXT_WINDOW)]
    os = torch.stack(os)  # stack turns our list of tensors into a 2d tensor
    for layer in layers:
        os = layer(os)
    loss = ((os - ys)**2).mean()

    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 5
    for p in parameters:
        p.data -= lr * p.grad

    if not j % (epochs // 10):
        print(f'{j} {loss.data=}')


for layer in layers:
    layer.training = False
with torch.no_grad():
    os = list(xs[:CONTEXT_WINDOW])
    for _ in range(t_size):
        input_os = torch.cat(tuple(os[-CONTEXT_WINDOW:]))
        input_os = input_os.unsqueeze(0)
        for layer in layers:
            input_os = layer(input_os)
        os.append(input_os[0])

    out_spec = []
    for f in range(f_size):
        f_slice = []
        for t in range(t_size):
            f_slice.append(complex(os[t][f],2) * freq_phase[f][t])
        out_spec.append(f_slice)

    f_reconstructed = librosa.istft(np.array(out_spec), n_fft=2048, hop_length=2048 // 2)
    soundfile.write('test_reconstruction_deep.wav', f_reconstructed, sr)

print('done')
