import numpy as np
import librosa
import soundfile
import torch
from torch import nn

datafile, sr = librosa.load('wawu.au', sr=None)
spec = librosa.stft(datafile, n_fft=2048, hop_length=2048 // 2)
freq_mag, freq_phase = librosa.magphase(spec)

f_size = len(freq_mag)
t_size = len(freq_mag[0])

data = []
for t in range(t_size):
    t_slice = [f[t] for f in freq_mag]
    data.append(t_slice)

xs = torch.Tensor(data[:-1])
ys = torch.Tensor(data[1:])

layers = [
    nn.Linear(f_size, f_size, bias=False),
    nn.BatchNorm1d(f_size),
]
parameters = [p for layer in layers for p in layer.parameters()]

os = []
epochs = 50000
for j in range(epochs):
    os = xs
    for layer in layers:
        os = layer(os)
    loss = ((os - ys)**2).mean()

    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 10
    for p in parameters:
        p.data -= lr * p.grad

    if not j % (epochs // 10):
        print(f'{j} {loss.data=}')


for layer in layers:
    layer.training = False
with torch.no_grad():
    os = [torch.tensor(xs[0])]
    for _ in range(t_size):
        input_os = os[-1].unsqueeze(0)
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
