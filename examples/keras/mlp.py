#!/usr/bin/env python

import sys, getopt, re, gzip, time
import numpy as np
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop

def mlp_data_read(fn):
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

def main(argv):
	n_hidden, n_epochs, minibatch, lr, heldout, seed, r_hidden, outfn, infn, use_multi_ce = 64, 20, 64, .001, 0.1, 11, 0.0, None, None, False

	def train_help():
		print("Usage: mlp.py [options] <input.knd> [output.knd]")
		print("Options:")
		print("  Model construction:")
		print("    -i FILE    load trained model from FILE []")
		print("    -o FILE    save trained model to FILE []")
		print("    -s INT     random seed [11]")
		print("    -n INT     number of hidden neurons [64]")
		print("    -d FLOAT   dropout rate at the hidden layers [0.0]")
		print("    -M         use multi-class cross-entropy")
		print("  Model training:")
		print("    -r FLOAT   learning rate [0.001]")
		print("    -v FLOAT   fraction of held-out data [0.0]")
		print("    -m INT     number of epochs [20]")
		print("    -B INT     minibatch size [64]")
		sys.exit(1)

	try:
		opts, args = getopt.getopt(argv[1:], "i:n:m:B:o:r:v:s:d:M")
	except getopt.GetoptError:
		train_help()
	if len(args) == 0:
		train_help()

	for opt, arg in opts:
		if opt == '-n': n_hidden = int(arg)
		elif opt == '-m': n_epochs = int(arg)
		elif opt == '-B': minibatch = int(arg)
		elif opt == '-i': infn = arg;
		elif opt == '-o': outfn = arg
		elif opt == '-r': lr = float(arg)
		elif opt == '-v': heldout = float(arg)
		elif opt == '-d': r_hidden = float(arg)
		elif opt == '-s': seed = int(arg)
		elif opt == '-M': use_multi_ce = True

	np.random.seed(seed)
	x, x_rnames, x_cnames = mlp_data_read(args[0])
	if len(args) >= 2: # training
		y, y_rnames, y_cnames = mlp_data_read(args[1])
		t_cpu = time.clock()
		model = Sequential()
		model.add(Dense(n_hidden, input_dim=len(x[0]), activation='relu'))
		if r_hidden > 0.0 and r_hidden < 1.0: model.add(Dropout(r_hidden))
		if use_multi_ce:
			model.add(Dense(len(y[0]), activation='softmax'))
			model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr), metrics=['accuracy'])
		else:
			model.add(Dense(len(y[0]), activation='sigmoid'))
			model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=lr), metrics=['accuracy'])
		model.fit(x, y, nb_epoch=n_epochs, batch_size=minibatch, validation_split=heldout)
		if outfn: model.save(outfn)
		sys.stderr.write("CPU time for training: " + str(time.clock() - t_cpu) + " sec\n")
	elif len(args) == 1 and infn:
		model = load_model(infn)
		y = model.predict(x)
		for i in range(len(y)):
			sys.stdout.write(x_rnames[i])
			for j in range(len(y[i])):
				sys.stdout.write("\t%g" % y[i][j])
			sys.stdout.write('\n')

if __name__ == "__main__":
	main(sys.argv)
