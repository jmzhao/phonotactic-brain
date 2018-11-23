# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for predicting phonetic scribes from itself.
Phonetic scribe: "/'wed\u026a\u014b/" ("wedding")
Padding is handled by using a repeated sentinel character (space)

Adapted from Keras example "addition_rnn.py"

Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.
'''

import sys
from keras.models import Sequential
# from keras.engine.training import slice_X
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent
import numpy as np
import json

if len(sys.argv) < 1 :
    sys.exit("Usage: python <script_name> <phonemic-scripts.jsonlines>"
        "* See /data/english_1000.jsonliones for example.")
INPUT_FILENAME = sys.argv[1]

import datetime, os
run_id = datetime.datetime.now().strftime('results/run_%Y-%m-%d_%H:%M:%S')
if not os.path.exists(run_id):
    os.makedirs(run_id)

class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


class Color:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

    @staticmethod
    def colorify(s, color=None, alarm=None) :
        if alarm is None and color is None :
            return s
        if color is None :
            color = Color.fail if alarm else Color.ok
        return color + s + Color.close

# Parameters for the model and dataset
INVERT = False ## invert the input script
LEN_LIMIT = 12
# Try replacing GRU, or SimpleRNN
RNN = recurrent.GRU
HIDDEN_SIZE = 64
BATCH_SIZE = 16
LAYERS = 1
EPOCHS = 20

def cleanse(script) :
    return script.translate({ord(c) : None for c in "'ˌ:/&+}"})##'ˌ/:

scripts = set()
print('Preparing data...')
with open(INPUT_FILENAME) as f :
    for line in f :
        data = json.loads(line)
        script = data.get("phonemic-script")
        if script is not None :
            script = cleanse(script)
            if len(script) <= LEN_LIMIT :
                scripts.add(script)
print('Total scripts:', len(scripts))

MAXLEN = max(map(len, scripts))
print('Maximum length:', MAXLEN)
scripts = [script + ' ' * (MAXLEN - len(script)) for script in scripts]

chars = ''.join(sorted(set(''.join(scripts))))
ctable = CharacterTable(chars, MAXLEN)
print('All symbols: "%s"'%chars)

print('Vectorization...')
X = np.zeros((len(scripts), MAXLEN, len(chars)), dtype=np.bool)
for i, script in enumerate(scripts):
    X[i] = ctable.encode(script[::-1] if INVERT else script, maxlen=MAXLEN)
y = np.zeros((len(scripts), MAXLEN, len(chars)), dtype=np.bool)
for i, script in enumerate(scripts):
    y[i] = ctable.encode(script, maxlen=MAXLEN)

# Shuffle (X, y) in unison as the later parts of X will almost all be larger digits
indices = np.arange(len(y))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Explicitly set apart 10% for validation data that we never train over
split_at = len(X) - len(X) // 10
(X_train, X_val) = (X[:split_at], X[split_at:])
(y_train, y_val) = (y[:split_at], y[split_at:])

print(X_train.shape)
print(y_train.shape)

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(MAXLEN))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributed(Dense(len(chars))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model each generation and show predictions against the validation dataset
model.save(run_id + '/model.h5')
open(run_id + '/model.json', 'w').write(model.to_json())
for iteration in range(1, EPOCHS):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=1,
              validation_data=(X_val, y_val))
    ###
    # Select 10 samples from the validation set at random so we can visualize errors
    for i in range(10):
        ind = np.random.randint(0, len(X_val))
        rowX, rowy = X_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowX, verbose=0)
        q = ctable.decode(rowX[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        # print('Q', q[::-1] if INVERT else q)
        print('T', correct)
        print(Color.colorify('☑', color=Color.ok) if correct == guess else Color.colorify('☒', color=Color.fail),
            ''.join(Color.colorify(g, alarm=(g!=c)) for g, c in zip(guess, correct)))
        print('---')

    model.save_weights(run_id + '/model.weights.epoch_%d.h5' % iteration)
