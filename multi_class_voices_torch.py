import numpy as np
import math
import matplotlib.pyplot as plt
import librosa
import cProfile
import random
import torch
import torch.nn.functional as F

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

min_len = min(len(sergei_data), len(sheep_data), len(andrej_data), 
              len(sergei2_data), len(sheep2_data), len(andrej2_data))

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

xs = torch.Tensor(np.array([x for x,_ in data]))
ys = torch.Tensor(np.array([label for _,label in data]))

txs = torch.Tensor(np.array([x for x,_ in test]))
tys = torch.Tensor(np.array([label for _,label in test]))

W = torch.randn((input_len, 3), requires_grad=True)
eps = 1e-8


for i in range(10000):
	# Forward pass
	# counts = (xs @ W).exp()
	# probs = counts/counts.sum(1, keepdims=True)
	# nll = -(probs*ys).sum(1, keepdims=True).log()
	# loss = nll.mean()
	# line below is effectively the same as the 4 lines above, but more efficient.
	loss = F.cross_entropy(xs @ W, ys)

	# Backward pass
	W.grad = None
	loss.backward()
	W.data -= 1 * W.grad


	# Sanity
	predicted = torch.argmax((txs @ W).exp(), dim=1)
	actual = torch.argmax((tys), dim=1)
	result = torch.eq(predicted,actual)
	accuracy = result.sum()/len(result)

	print(f'{loss.data=}, {accuracy=}')