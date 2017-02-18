#!/usr/bin/env python

import sys, getopt, re, gzip
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model
from keras.optimizers import RMSprop

def sann_data_read(fn):
	x, row_names, col_names = [], [], []

	def _process_fp(fp):
		for l in fp:
			t = l[:-1].split('\t')
			if l[0] == '#':
				col_names = t[1:]
			else:
				row_names.append(t[0])
				x.append(t[1:]);

	if re.search(r'\.gz$', fn):
		with gzip.open(fn, 'r') as fp:
			_process_fp(fp)
	else:
		with open(fn, 'r') as fp:
			_process_fp(fp)
	return np.array(x).astype('float32'), row_names, col_names

def main_train(argv):
	n_hidden, n_epochs, minibatch, lr, heldout, seed, r_hidden, outfn = 50, 20, 64, .001, .1, 11, 0.0, None

	def train_help():
		print("Usage: mlp.py train [options] <input.snd> <output.snd>")
		print("Options:")
		print("  Model construction:")
		print("    -h INT     number of hidden neurons [50]")
		print("    -s INT     random seed [11]")
		print("    -o FILE    save model to FILE []")
		print("  Model training:")
		print("    -e FLOAT   learning rate [0.001]")
		print("    -T FLOAT   fraction of held-out data [0.1]")
		print("    -R FLOAT   dropout rate at the hidden layers [0.0]")
		print("    -n INT     number of epochs [20]")
		print("    -B INT     minibatch size [64]")
		sys.exit(1)

	try:
		opts, args = getopt.getopt(argv, "h:n:B:o:e:T:s:R:")
	except getopt.GetoptError:
		train_help()
	if len(args) < 2:
		train_help()

	for opt, arg in opts:
		if opt == '-h': n_hidden = int(arg)
		elif opt == '-n': n_epochs = int(arg)
		elif opt == '-B': minibatch = int(arg)
		elif opt == '-o': outfn = arg
		elif opt == '-e': lr = float(arg)
		elif opt == '-T': heldout = float(arg)
		elif opt == '-R': r_hidden = float(arg)
		elif opt == '-s': seed = int(arg)

	np.random.seed(seed)
	x, x_rnames, x_cnames = sann_data_read(args[0])
	y, y_rnames, y_cnames = sann_data_read(args[1])
	model = Sequential()
	model.add(Dense(n_hidden, input_dim=len(x[0]), activation='relu'))
	if r_hidden > 0.0 and r_hidden < 1.0: model.add(Dropout(r_hidden))
	model.add(Dense(len(y[0]), activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=lr), metrics=['accuracy'])
	model.fit(x, y, nb_epoch=n_epochs, batch_size=minibatch, validation_split=heldout)
	if outfn: model.save(outfn)

def main_apply(argv):
	if len(argv) < 2:
		print("Usage: mlp.py apply <model> <input.snd>")
		sys.exit(1)
	model = load_model(argv[0])
	x, x_rnames, x_cnames = sann_data_read(argv[1])
	y = model.predict(x)
	for i in range(len(y)):
		sys.stdout.write(x_rnames[i])
		for j in range(len(y[i])):
			sys.stdout.write("\t%g" % y[i][j])
		sys.stdout.write('\n')

def main(argv):
	if len(argv) < 2:
		print("Usage: sann-keras <command> <arguments>")
		print("Command:")
		print("  train     train the model")
		print("  apply     apply the model")
		sys.exit(1)
	elif argv[1] == 'train':
		main_train(argv[2:])
	elif argv[1] == 'apply':
		main_apply(argv[2:])

if __name__ == "__main__":
	main(sys.argv)
