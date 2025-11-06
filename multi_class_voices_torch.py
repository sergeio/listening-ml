import numpy as np
import matplotlib.pyplot as plt
import librosa
import cProfile
import random
import torch

VOLUME_THRESHOLD = 8
andrej_raw, _ = librosa.load('andrej.au', sr=None)
sergei_raw, _ = librosa.load('sergei_voice.au', sr=None)
sheep_raw, _ = librosa.load('sheep_voice.au', sr=None)

andrej2_raw, _ = librosa.load('andrej2.au', sr=None)
sergei2_raw, _ = librosa.load('sergei2.au', sr=None)
sheep2_raw, _ = librosa.load('sheep2.au', sr=None)

def is_silence(f):
    return max(f) < VOLUME_THRESHOLD

def make_training_data(y, label):
    data = []
    spec = librosa.stft(y, n_fft=2048, hop_length=2048)
    for t in range(len(spec[0])):
        t_slice = [abs(x[t]) for x in spec]
        t_slice = t_slice[2:200]
        t_slice = [sum(t_slice[5*i:5*i+5]) for i in range(int(len(t_slice) / 5))]

        if not is_silence(t_slice):
            t_slice = np.array(t_slice)
            t_slice = t_slice / (np.max(t_slice))
            data.append(t_slice)

    return [(x, label) for x in data]

sergei_data = make_training_data(sergei_raw, [1, 0, 0])
sheep_data = make_training_data(sheep_raw, [0, 1, 0])
andrej_data = make_training_data(andrej_raw, [0, 0, 1])

sergei2_data = make_training_data(sergei2_raw, [1, 0, 0])
sheep2_data = make_training_data(sheep2_raw, [0, 1, 0])
andrej2_data = make_training_data(andrej2_raw, [0, 0, 1])

print('sergei_data', len(sergei_data))
print('sheep_data', len(sheep_data))
print('andrej_data', len(andrej_data))
min_len = min(len(sergei_data), len(sheep_data), len(andrej_data),
              len(sergei2_data), len(sheep2_data), len(andrej2_data))
print('min_len', min_len)

data = sergei_data[:min_len] + sheep_data[:min_len] + andrej_data[:min_len]
data2 = sergei2_data[:min_len] + sheep2_data[:min_len] + andrej2_data[:min_len]
data = data + data2
print('data', len(data))
random.shuffle(data)


cutoff = int(len(data) * .5)
data, test = data[:cutoff], data[cutoff:]

input_len = len(data[0][0])
print('input_len', input_len)



# Pytorch stuff

xs = torch.Tensor([x for x,_ in data])
ys = torch.Tensor([label for _,label in data])

txs = torch.Tensor([x for x,_ in test])
tys = torch.Tensor([label for _,label in test])

V = torch.randn((input_len, input_len), requires_grad=True)
W = torch.randn((input_len, 3), requires_grad=True)
eps = 1e-8

output = []
for i in range(int(1e4)):
    # Forward pass
    counts = (xs @ V).exp()
    probs = counts/counts.sum(1, keepdims=True)
    counts = (probs @ W).exp()
    probs = counts/counts.sum(1, keepdims=True)
    nll = -(probs*ys).sum(1, keepdims=True).log()
    loss = nll.mean()
    # Backward pass
    V.grad = None
    W.grad = None
    loss.backward()
    W.data -= 300 * W.grad

    # Sanity
    test_predicted = torch.argmax((txs @ V @ W).exp(), dim=1)
    test_actual = torch.argmax((tys), dim=1)
    test_result = torch.eq(test_predicted,test_actual)
    test_accuracy = test_result.sum()/len(test_result)

    data_predicted = torch.argmax((xs @ V @ W).exp(), dim=1)
    data_actual = torch.argmax((ys), dim=1)
    data_result = torch.eq(data_predicted,data_actual)
    data_accuracy = data_result.sum()/len(data_result)

    print(f'{loss.data=} {data_accuracy=} {test_accuracy=}')

print('\n'.join(output[:20]))
print()
print('\n'.join(output[-10:]))
