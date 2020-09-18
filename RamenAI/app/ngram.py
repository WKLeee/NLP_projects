import json
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Embedding
import data_processing as dp
import config as cg
import sequencer
from numpy import array
import time
import argparse

DEFAULT_N = 2
DEFAULT_EPOCHS=20
DEFAULT_OUT='model_{0}.h5'.format(time.strftime("%Y%m%d-%H%M%S"))

parser = argparse.ArgumentParser('populate')
parser.add_argument('--model', dest='model', type=str, help='the model to load')
parser.add_argument('--epochs', dest='epochs', type=int, default=DEFAULT_EPOCHS, help='number of epochs to use while training')
parser.add_argument('--stars', dest='stars', type=int, help='number of stars')
parser.add_argument('--out', dest='out', type=str, default=DEFAULT_OUT, help='the path name for the saved model')
parser.add_argument('--limit', dest='limit', type=int, help='limit to number of reviews used')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.add_argument('--n', dest='n', type=int, default=DEFAULT_N, help='size of ngrams')
args = parser.parse_args()

# generate a sequence from the model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# pre-pad sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
	return in_text

data = dp.load_file(cg.MINI_DATA_PATH)

if args.stars is not None:
	stars = int(args.stars)
	data = list(filter(lambda review : int(review['stars']) == stars, data))

if args.limit is not None:
	data = data[0:args.limit]

print('using {0} reviews'.format(len(data)))

reviews = [review['text'] for review in data]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)
encoded = tokenizer.texts_to_sequences(reviews)

vocab_size = len(tokenizer.word_index) + 1
print('vocab size = {0}'.format(vocab_size))

print('processing {0}_grams'.format(args.n))
sequences = list()
for line in encoded:
    sequences.extend(sequencer.sequence_line(line, args.n))

sequences = array(sequences)
X, y = sequences[:,0:-1], sequences[:,-1]
# one hot encoding
#y = to_categorical(y, num_classes=vocab_size)

model = None

if args.model is None:
	model = Sequential()
	model.add(Embedding(input_dim=vocab_size, output_dim=10, input_length = args.n - 1))
	if args.gpu is True:
		model.add(CuDNNLSTM(50))
	else :
		model.add(LSTM(50))
	model.add(Dense(vocab_size, activation='softmax'))
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	# fit network
	model.fit(X, y, epochs=args.epochs, verbose=2, batch_size=256)
	# Creates a HDF5 file 'my_model.h5'
	model.save(args.out)
	print('saved model to file {0}'.format(args.out))
else:
	model = load_model(args.model)

print(generate_seq(model, tokenizer, args.n - 1, 'the food is not', 24))
print(generate_seq(model, tokenizer, args.n - 1, 'food on the table', 24))
print(generate_seq(model, tokenizer, args.n - 1, 'we had a menu', 24))
