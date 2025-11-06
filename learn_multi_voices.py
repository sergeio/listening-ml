import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
import librosa
import cProfile
import random
import math

from engine import Value
from nn import Neuron, Layer, MLP

VOLUME_THRESHOLD = 8
andrej_raw, _ = librosa.load('andrej.au', sr=None)
sergei_raw, _ = librosa.load('sergei_voice.au', sr=None)
sheep_raw, _ = librosa.load('sheep_voice.au', sr=None)

andrej2_raw, _ = librosa.load('andrej2.au', sr=None)
sergei2_raw, _ = librosa.load('sergei2.au', sr=None)
sheep2_raw, _ = librosa.load('sheep2.au', sr=None)


def normalize(y):
    return np.array([x/max(y) for x in y])
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
            data.append(t_slice)

        t_slice = np.array(t_slice)
        t_slice = t_slice / (np.max(t_slice))# + 1e-8)
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

# # print(data[0])
# for d in data[:20]:
#   plt.plot(d[0])
#   plt.show()


cutoff = int(len(data) * .5)
data, test = data[:cutoff], data[cutoff:]

input_len = len(data[0][0])
print('input_len', input_len)
model = MLP(input_len, [input_len, 3])
for p in model.parameters():
    p.data *= 0.2

# model = MLP(input_len, [input_len, input_len, 1])
# model = MLP(200, [100, 100, 1])

print('cutoff', cutoff)
print('length of  data: ',len(data))
print('length of test data: ',len(test))

def predict(test_data):
    accuracy = 0.0
    for item, label in test_data:
        prediction = model(item)
        prediction = softmax(prediction)
        # prediction is a list of 3 Value objects
        # Find the index with the highest prediction value
        pred_values = [p.data for p in prediction]
        pred_index = pred_values.index(max(pred_values))
        # Find the index with 1 in the label
        label_index = label.index(1)
        if pred_index == label_index:
            accuracy += 1
    accuracy /= len(test_data)
    return accuracy

def softmax(values):
    exps = [v.exp() for v in values]
    s = sum(exps)
    return [e / s for e in exps]

def loss(data, model):
    eps = 1e-8
    total = Value(0.0)
    accuracy = 0.0

    for x, label in data:
        correct = False
        out = model(x)
        probs = softmax(out)
        sample_loss = -sum(label[i] * (probs[i] + eps).log() for i in range(3))
        prob_values = [v.data for v in probs]
        if prob_values.index(max(prob_values)) == label.index(1):
            accuracy += 1
            correct = True
        # if random.random() < 1e-2:
        #     print(f'{correct=} {prob_values=} {label=}')

        total = total + sample_loss

    # This is a Value object division
    return total / len(data), accuracy / len(data)

# def softmax(values):
#     exps = [np.exp(v.data) for v in values]
#     s = sum(exps)
#     return [Value(e / s) for e in exps]

# def loss(data, model):
#     eps = 1e-8
#     total = 0
#     for x, label in data:
#         out = model(x)
#         probs = softmax(out)
#         total += -sum(label[i] * np.log(probs[i].data + eps) for i in range(3))
#     return Value(total / len(data))

# def loss(data, model):
#     loss = 0.0
#     results = [softmax(model(input_)) for input_, _ in data]
#     #results = softmax(results)
#     labels = [label for _,label in data]
#
#     for result, label in zip(results, labels):
#         # result is a list of 3 Value objects, label is [1,0,0] or [0,1,0] or [0,0,1]
#         # Sum of squared differences
#         for i in range(3):
#             loss += (result[i] - label[i])**2
#
#     loss = loss/len(data)
#
#     # l2 = .001 * sum([p * p for p in model.parameters()]) / len(model.parameters())
#     return loss


for i in range(15):
    total_loss, data_accuracy = loss(data, model)

    model.zero_grad()
    total_loss.backward()
    grads = [abs(p.grad) for p in model.parameters()]

    learning_rate = .01
    #learning_rate = .1 - .9*i/100
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    # test_accuracy = predict(data)
    print(i, 'loss=', total_loss.data, 'dacc=', data_accuracy)
