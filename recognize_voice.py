import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa
import cProfile
import random

from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP

VOLUME_THRESHOLD = 8
#raw, sr = librosa.load('scale.au', sr=None)
sergei_raw, sergei_sr = librosa.load('sergei_voice.au', sr=None)
sheep_raw, sheep_sr = librosa.load('sheep_voice.au', sr=None)

sergei2_raw, sergei2_sr = librosa.load('sergei2.au', sr=None)
sheep2_raw, sheep2_sr = librosa.load('sheep2.au', sr=None)

def normalize(y):
    return np.array([x/max(y) for x in y])
def is_silence(f):
    return max(f) < VOLUME_THRESHOLD

def make_training_data(y, sr, label):
    data = []
    spec = librosa.stft(y, n_fft=2048, hop_length=2048)
    for t in range(len(spec[0])):
        t_slice = [abs(x[t]) for x in spec]
        t_slice = t_slice[2:200]
        t_slice = [sum(t_slice[5*i:5*i+5]) for i in range(int(len(t_slice) / 5))]
        if not is_silence(t_slice):
            data.append(t_slice)
    return [(x, label) for x in data]

sergei_data = make_training_data(sergei_raw[2048*16:2048*256], sergei_sr, -1.0)
sheep_data = make_training_data(sheep_raw[2048*16:2048*256], sheep_sr, 1.0)
sergei2_data = make_training_data(sergei2_raw[2048*16:2048*256], sergei2_sr, -1.0)
sheep2_data = make_training_data(sheep2_raw[2048*16:2048*256], sheep2_sr, 1.0)

print('sergei_data', len(sergei_data))
print('sheep_data', len(sheep_data))
min_len = min(len(sergei_data), len(sheep_data))
data = sergei_data[:min_len] + sheep_data[:min_len]
print('data', len(data))
random.shuffle(data)

# # print(data[0])
# for d in data[:20]:
#   plt.plot(d[0])
#   plt.show()


cutoff = int(len(data) * .5)
data, test = data[:cutoff], data[cutoff:]

input_len = len(data[0][0])
print('input_len', input_len)
# model = MLP(input_len, [input_len, input_len, 1])
# model = MLP(200, [100, 100, 1])

print('cutoff', cutoff)
print('length of  data: ',len(data))
print('length of test data: ',len(test))

# model = pickle.load(open('model52.pkl', 'rb'))
model = pickle.load(open('model18.pkl', 'rb'))
print(model)

def predict(test_data):
    accuracy = 0.0
    for item, label in test_data:
        prediction = model(item)
        if (prediction.data > 0) == (label > 0):
            accuracy += 1
    accuracy /= len(test_data)
    return accuracy

def loss(data):
    loss = 0.0
    results = [model(input_) for input_, _ in data]
    labels = [label for _,label in data]

    for result, label in zip(results, labels):
        # loss += (label-result).relu()
        loss += (1 + -label*result).relu()
    loss = loss/len(data)

    return loss


accuracy = predict(sergei_data)
print('sergei1 acc=', accuracy)

accuracy = predict(sergei2_data)
print('sergei2 acc=', accuracy)

accuracy = predict(sheep_data)
print('sheep1 acc=', accuracy)

accuracy = predict(sheep2_data)
print('sheep2 acc=', accuracy)
