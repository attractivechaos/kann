#!/usr/bin/env python

import sys, getopt, os, re, gzip, time
import numpy as np
import tensorflow as tf

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

def mlp_model_gen(n_in, n_out, n_layer, n_hidden, use_multi_ce):
	t = tf.placeholder(tf.float32, [None, n_in], name="in")
	for i in range(n_layer):
		t = tf.layers.dense(t, n_hidden, activation=tf.nn.relu)
	t = tf.layers.dense(t, n_out)
	out = tf.nn.softmax(t, name="out")
	truth = tf.placeholder(tf.float32, [None, n_out], name="truth")
	if use_multi_ce: t = tf.nn.softmax_cross_entropy_with_logits(logits=t, labels=truth)
	else: t = tf.nn.sigmoid_cross_entropy_with_logits(logits=t, labels=truth)
	cost = tf.reduce_mean(t, name="cost")
	return cost

def main(argv):
	n_layer, n_hidden, max_epoch, minibatch, lr, seed, r_hidden, outdir, indir, use_multi_ce = 1, 64, 20, 64, .001, 11, 0.0, None, None, False
	n_threads = 1

	def train_help():
		print("Usage: mlp.py [options] <input.knd> [output.knd]")
		print("Options:")
		print("  Model construction:")
		print("    -i DIR     load trained model from DIR []")
		print("    -o DIR     save trained model to DIR []")
		print("    -s INT     random seed [11]")
		print("    -l INT     number of hidden layers [1]")
		print("    -n INT     number of hidden neurons per layer [64]")
		print("    -d FLOAT   dropout at the hidden layer(s) [0.0]")
		print("    -M         use multi-class cross-entropy")
		print("  Model training:")
		print("    -r FLOAT   learning rate [0.001]")
		print("    -m INT     number of epochs [20]")
		print("    -B INT     minibatch size [64]")
		sys.exit(1)

	try:
		opts, args = getopt.getopt(argv[1:], "n:m:B:i:o:r:s:d:l:Mt:")
	except getopt.GetoptError:
		train_help()
	if len(args) < 1:
		train_help()

	for opt, arg in opts:
		if opt == '-n': n_hidden = int(arg)
		elif opt == '-l': n_layer = int(arg)
		elif opt == '-m': max_epoch = int(arg)
		elif opt == '-B': minibatch = int(arg)
		elif opt == '-i': indir = arg
		elif opt == '-o': outdir = arg
		elif opt == '-r': lr = float(arg)
		elif opt == '-d': r_hidden = float(arg)
		elif opt == '-s': seed = int(arg)
		elif opt == '-M': use_multi_ce = True
		elif opt == '-t': n_threads = int(arg)

	tf.set_random_seed(seed)
	sys.stderr.write("Reading input...\n")
	x_dat, x_rnames, x_cnames = mlp_data_read(args[0])

	conf = tf.ConfigProto(intra_op_parallelism_threads=n_threads, inter_op_parallelism_threads=n_threads)
	if len(args) >= 2: # training
		sys.stderr.write("Reading truth...\n")
		y_dat, y_rnames, y_cnames = mlp_data_read(args[1])

		sys.stderr.write("Training...\n")
		t_cpu = time.clock()
		t_real = time.time()
		cost = mlp_model_gen(len(x_dat[0]), len(y_dat[0]), n_layer, n_hidden, use_multi_ce)
		optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

		with tf.Session(config=conf) as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(max_epoch):
				off, tot_cost = 0, 0
				while off < len(x_dat):
					mb = minibatch
					if mb > len(x_dat) - off: mb = len(x_dat) - off
					xb, yb = x_dat[off:off+mb], y_dat[off:off+mb]
					_, c = sess.run([optimizer, cost], { "in:0":xb, "truth:0":yb })
					tot_cost += c
					off += mb
				avg_cost = tot_cost / len(x_dat)
				sys.stderr.write("epoch: {}; cost: {:.6f}\n".format(epoch+1, avg_cost))

			if outdir:
				if outdir and not os.path.isdir(outdir): os.mkdir(outdir)
				saver = tf.train.Saver()
				saver.save(sess, outdir + "/model")

		sys.stderr.write("CPU time for training: {:.2f}\n".format(time.clock() - t_cpu))
		sys.stderr.write("Real time for training: {:.2f}\n".format(time.time() - t_real))
	elif len(args) == 1 and indir: # prediction
		with tf.Session(config=conf) as sess:
			saver = tf.train.import_meta_graph(indir + "/model.meta")
			saver.restore(sess, tf.train.latest_checkpoint(indir))
			out = tf.get_default_graph().get_tensor_by_name("out:0")
			y_dat = out.eval({ "in:0":x_dat })
			for i in range(len(x_dat)):
				print('{}\t{}'.format(x_rnames[i], "\t".join(map(str, y_dat[i]))))

if __name__ == "__main__":
	main(sys.argv)
