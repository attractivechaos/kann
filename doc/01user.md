## Table of Contents

* [Overview](#overview)
* [Constructing a Neural Network](#constructing-a-neural-network)
  - [Constructing a multi-layer perceptron (MLP)](#constructing-a-multi-layer-perceptron-mlp)
  - [Constructing a convolutional neural network (CNN)](#constructing-a-convolutional-neural-network-cnn)
  - [Constructing a denoising autoencoder (dAE) with tied weights](#constructing-a-denoising-autoencoder-dae-with-tied-weights)
  - [Constructing a recurrent neural network (RNN)](#constructing-a-recurrent-neural-network-rnn)
* [Training a Neural Network](#training-a-neural-network)
  - [Training a simple feedforward neural network (FNN)](#training-a-simple-feedforward-neural-network-fnn)
  - [Training a recurrent neural network (RNN)](#training-a-recurrent-neural-network-rnn)



## Overview

In KANN, every neural network is modeled by a computation graph. A computation
graph represents one or multiple mathematical expressions. It is a directed
acyclic graph, where an external node represents a constant or a variable used
in the expressions; an internal node represents an operator (e.g. plus) or a
function (e.g. exponential); an edge from node `u` to `v` indicates `u` being
an operand of `v`.

Files `kautodiff.*` implement computation graphs and symbol-to-number
reverse-mode automatic differentiation. Users construct a graph by composing
math expressions with operators defined in the library, and then use the graph
to compute values and partial derivatives of a scalar function. With
`kautodiff.*`, users are required to explicitly define and manage every node.
Files `kann.*` simplify this procedure. This part provides layers that can
specify multiple nodes at the same time by not exposing variables associated
with one or multiple operators. A program is expected to call both
`kautodiff.*` and `kann.*` APIs.



## Constructing a Neural Network

In KANN, a neural network is essentially a computational graph. Constructing a
neural network boils down to constructing a directed acyclic computational
graph.

### Constructing a multi-layer perceptron (MLP)

The following function constructs an MLP with one hidden layer.
```c
kann_t *model_gen(int n_in, int n_out, int n_hidden_neurons)
{
	kad_node_t *t;
	t = kann_layer_input(n_in);
	t = kann_layer_linear(t, n_hidden_neurons);
	t = kad_relu(t);
	t = kann_layer_cost(t, n_out, KANN_C_CEB);
	return kann_new(t, 0);
}
```
Here `kann_layer_input()` sets an input node in the computational graph.
`kann_layer_linear()` adds a linear transformation layer to the graph and
`kad_relu()` sets the activation function. `kann_layer_cost()` adds an output
layer and a binary cross-entropy cost (specified by `KANN_C_CEB`). Finally
`kann_new()` generates the neural network.

### Constructing a convolutional neural network (CNN)

The following function constructs a CNN to classify MNIST images:
```c
kann_t *model_gen_mnist(int n_h_flt, int n_h_fc)
{
	kad_node_t *t;
	t = kad_feed(4, 1, 1, 28, 28), t->ext_flag |= KANN_F_IN;
	t = kad_relu(kann_layer_conv2d(t, n_h_flt, 3, 3, 1, 0));
	t = kad_relu(kann_layer_conv2d(t, n_h_flt, 3, 3, 1, 0));
	t = kann_layer_max2d(t, 2, 2, 2, 0);
	t = kann_layer_dropout(t, 0.2f);
	t = kann_layer_linear(t, n_h_fc);
	t = kad_relu(t);
	return kann_new(kann_layer_cost(t, 10, KANN_C_CEM), 0);
}
```
It uses a little more low-level APIs. Here we use `kad_feed()` to add an input
node and set an external flag `KANN_F_IN` to mark it. The input is a 4D array
with the four dimensions being: mini-batch size, number of channels, height and
width. **Importantly**, we note that the first dimension of input, truth and
most of internal nodes in a neural network is always the mini-batch size.
Violating this rule might lead to unexpected errors. The rest of code adds two
convolution layers and one max pooling layer. We are using multi-class
cross-entropy cost (specified by `KANN_C_CEM`) in this example.

### Constructing a denoising autoencoder (dAE) with tied weights

The following function demonstrates how to use shared weights.
```c
kann_t *model_gen(int n_in, int n_hidden, float i_dropout)
{
	kad_node_t *x, *t, *w;
	w = kann_new_weight(n_hidden, n_in);
	x = kad_feed(2, 1, n_in), x->ext_flag |= KANN_F_IN | KANN_F_TRUTH;
	t = kann_layer_dropout(x, i_dropout);
	t = kad_tanh(kad_add(kad_cmul(t, w), kann_new_bias(n_hidden)));
	t = kad_add(kad_matmul(t, w), kann_new_bias(n_in));
	t = kad_sigm(t), t->ext_flag = KANN_F_OUT;
	t = kad_ce_bin(t, x), t->ext_flag = KANN_F_COST;
	return kann_new(t, 0);
}
```
In this example, the input node is also marked as the truth node. The weight
matrix `w`, with `n_hidden` rows and `n_in` columns, is first used at the
encoding phase in `kad_cmul()` (matrix product with the second matrix
transposed) and then used again at the decoding phase in `kad_matmul()` (matrix
product). The input node is also reused to compute the cost.

Generally, to use a shared variable, we keep the pointer to the variable node
and use it in multiple expressions. This procedure often requires to interact
with low-level `kad_*` APIs, as `kann_layer_*` APIs hide variables.

### Constructing a recurrent neural network (RNN)

The following function constructs an RNN with one GRU unit. It has a sequence
of input and a sequence of output. Such a model may be used for character-level
text generation.
```c
kann_t *model_gen(int n_in, int n_out, int n_hidden)
{
	kad_node_t *t;
	t = kann_layer_input(n_in);
	t = kann_layer_gru(t, n_hdden, 0);
	return kann_new(kann_layer_cost(t, n_out, KANN_C_CEB), 0);
}
```
When classify a sequence, we would like the network to have one output, instead
of a sequence of output. We can construct the network this way:
```c
kann_t *model_gen(int n_in, int n_out, int n_hidden)
{
	kad_node_t *t;
	t = kann_layer_input(n_in);
	t = kann_layer_gru(t, n_hdden, 0);
	t = kad_avg(1, &t);
	return kann_new(kann_layer_cost(t, n_out, KANN_C_CEB), 0);
}
```
This model averages the hidden output from GRU and then apply a linear layer to
derive the final output (done by `kann_layer_cost()`).



## Training a Neural Network

### Training a simple feedforward neural network (FNN)

If an FNN only has one input node and one output node, we can use the
`kann_train_fnn1()` API to train it. The API uses RMSprop for minimization. It
splits data into training and validation data and stops training until the
validation accuracy is not improved after, say, 10 epochs.

The `kann_train_fnn1()` function is relatively short. We encourage users to
read this function to understand its internals. When the network has multiple
inputs or outputs, or when we want to use another training policy,
`kann_train_fnn1()` would not work any more; we may have to roll our own
training code.

### Training a recurrent neural network (RNN)

The KANN computational graph does not keep the history of computation. To train
an RNN, we have to unroll it with `kann_unroll()`. Variables and constants are
shared between the original and the unrolled networks. Training the unrolled
network simultaneously trains the original network. As the unrolled network has
multiple input nodes, we cannot use `kann_train_fnn1()` for training.  We are
not providing a `kann_train_fnn1()` like API because converting all time series
data to vectors may take too much memory (for example, converting text to
vectors at the character level makes the input 1000 times larger). We tried a
callback-based API in an older version of KANN, but found it is confusing to
use and is not flexible enough.

For now, the only way to train an RNN is to manually write our own training
routine. The following example shows how to train an RNN for character-level
text generation:
```c
void train(kann_t *ann, float lr, int ulen, int mbs, int max_epoch, int len, const uint8_t *data)
{
	int i, k, n_var, n_char;
	float **x, **y, *r, *g;
	kann_t *ua;

	n_char = kann_dim_in(ann);
	x = (float**)calloc(ulen, sizeof(float*)); // an unrolled has _ulen_ input nodes
	y = (float**)calloc(ulen, sizeof(float*)); // ... and _ulen_ truth nodes
	for (k = 0; k < ulen; ++k) {
		x[k] = (float*)calloc(n_char, sizeof(float)); // each input node takes a (1,n_char) 2D array
		y[k] = (float*)calloc(n_char, sizeof(float)); // ... where 1 is the mini-batch size
	}
	n_var = kann_size_var(ann);               // total size of variables
	r = (float*)calloc(n_var, sizeof(float)); // temporary array for RMSprop
	g = (float*)calloc(n_var, sizeof(float)); // gradients

	ua = kann_unroll(ann, ulen);            // unroll; the mini batch size is 1
	kann_feed_bind(ua, KANN_F_IN,    0, x); // bind _x_ to input nodes
	kann_feed_bind(ua, KANN_F_TRUTH, 0, y); // bind _y_ to truth nodes
	for (i = 0; i < max_epoch; ++i) {
		double cost = 0.0;
		int j, b, tot = 0, n_cerr = 0;
		for (j = 1; j + ulen * mbs - 1 < len; j += ulen * mbs) {
			memset(g, 0, n_var * sizeof(float));
			for (b = 0; b < mbs; ++b) { // loop through a mini-batch
				for (k = 0; k < ulen; ++k) {
					memset(x[k], 0, n_char * sizeof(float));
					memset(y[k], 0, n_char * sizeof(float));
					x[k][data[j+b*ulen+k-1]] = 1.0f;
					y[k][data[j+b*ulen+k]] = 1.0f;
				}
				cost += kann_cost(ua, 0, 1) * ulen;
				n_cerr += kann_class_error(ua);
				tot += ulen;
				for (k = 0; k < n_var; ++k) g[k] += ua->g[k];
			}
			for (k = 0; k < n_var; ++k) g[k] /= mbs; // gradients are the average of this mini batch
			kann_RMSprop(n_var, lr, 0, 0.9f, g, ua->x, r); // update all variables
		}
		fprintf(stderr, "epoch: %d; cost: %g (class error: %.2f%%)\n", i+1, cost / tot, 100.0 * n_cerr / tot);
	}
	kann_delete_unrolled(ua); // for an unrolled network, don't use kann_delete()!

	for (k = 0; k < ulen; ++k) { free(x[k]); free(y[k]); }
	free(g); free(r); free(y); free(x);
}
```
