## Getting Started
```sh
# acquire source code and compile
git clone https://github.com/attractivechaos/kann
cd kann; make
# learn unsigned addition (30000 samples; numbers within 10000)
seq 30000 | awk -v m=10000 '{a=int(m*rand());b=int(m*rand());print a,b,a+b}' \
  | ./examples/rnn-bit -m5 -o add.kan -
# apply the model (output 1138429, the sum of the two numbers)
echo 400958 737471 | ./examples/rnn-bit -Ai add.kan -
# learn multiplication to a number smaller than 100:
seq 30000 | awk '{a=int(10000*rand());b=int(100*rand())+1;print a,b,a*b}' \
  | ./examples/rnn-bit -m30 -l2 -n128 -o mul100.kan -
# apply the model to large numbers (answer: 1486734150878261153)
echo 15327156194621249 97 | ./examples/rnn-bit -Ai 1.kan -
```

## Introduction

KANN is a standalone 4-file library in C for constructing and training
small to medium artificial neural networks such as [MLP][mlp], [CNN][cnn] and
[LSTM][lstm]. It implements generic reverse-mode [automatic
differentiation][ad] based on the concept of computational graph and allows to
construct topologically complex neural networks with shared weights, multiple
inputs/outputs and recurrence. KANN is flexible, portable, small and fairly
efficient for its size.

### Background and motivations

Mainstream deep learning frameworks often consist of over 100k lines of code by
itself and additionally have multiple non-trivial dependencies such as Python,
BLAS, HDF5 and ProtoBuf. This makes it hard to deploy on older machines and
difficult for general programmers to understand the internals. While there are
several lightweight frameworks, they are still fairly heavy and lack important
features (e.g. RNN) and flexibility (e.g. arbitrary weight sharing) of
mainstream frameworks.  It is non-trivial and often impossible to use these
lightweight frameworks to construct non-standard neural networks.

We developed KANN, 1) to fully understand the algorithms behind mainstream
frameworks; 2) to have a foundation flexible enough to experiment our own
small but contrived models; 3) to give other C/C++ programmers a tiny and
efficient library that can be easily integrated into their tools without
worrying about [dependency hell][dh].

### Features

* Flexible. Model construction by building a computational graph with
  operators. Support RNNs, weight sharing and multiple inputs/outputs.

* Reasonably efficient. Support mini-batching. Optimized matrix product and
  convolution, coming close to (though not as fast as) OpenBLAS and mainstream
  deep learning frameworks. KANN may optionally work with BLAS libraries,
  enabled with the `HAVE_CBLAS` macro.

* Small. As of now, KANN has less than 3000 coding lines in four source code
  files, with no non-standard dependencies by default.

### Limitations

* CPU only; no parallelization. KANN does not support GPU or multithreading for
  now. As such, KANN is **not** intended for training huge neural networks.

* No bidirectional RNN (achievable by manually unrolling RNN, but tedious and
  not general enough). No batch normalization.

* Verbose APIs for training RNNs.

## Documentations

Comments in the header files briefly explain the APIs. More documentations can
be found in the [doc](doc) directory. Examples using the library are in the
[examples](examples) directory.

### A tour of basic KANN APIs

Working with neural networks usually involves three steps: model construction,
training and prediction. We can use layer APIs to build a simple model:
```c
kann_t *ann;
kad_node_t *t;
t = kann_layer_input(784); // for MNIST
t = kad_relu(kann_layer_linear(t, 64)); // a 64-neuron hidden layer with ReLU activation
t = kann_layer_cost(t, 10, KANN_C_CEM); // softmax output + multi-class cross-entropy cost
ann = kann_new(t, 0);                   // compile the network and collate variables
```
For this simple feedforward model with one input and one output, we can train
it with:
```c
int n;     // number of training samples
float **x; // model input, of size n * 784
float **y; // model output, of size n * 10
// fill in x and y here and then call:
kann_train_fnn1(ann, 0.001f, 64, 25, 10, 0.1f, n, x, y);
```
We can save the model to a file with `kann_save()` or use it to classify a
MNIST image:
```c
float *x;       // of size 784
const float *y; // this will point to an array of size 10
// fill in x here and then call:
y = kann_apply1(ann, x);
```

Working with complex models requires to use low-level APIs. Please see
[01user.md](doc/01user.md) for details.

### A complete example

This example learns addition between two 10-bit integers:
```c
// to compile and run: gcc -O2 this-prog.c kann.c kautodiff.c && ./a.out
#include <stdlib.h>
#include <stdio.h>
#include "kann.h"

int main(void)
{
	int i, k, max_bit = 10, n_samples = 30000, mask, n_err;
	kann_t *ann;
	float **x, **y;
	kad_node_t *t;
	// construct an MLP with two hidden layers
	t = kann_layer_input(max_bit * 2); // two numbers in binary representation
	t = kad_relu(kann_layer_linear(t, 64));
	t = kad_relu(kann_layer_linear(t, 64));
	t = kann_layer_cost(t, max_bit * 2, KANN_C_CEB); // output uses 1-hot encoding
	ann = kann_new(t, 0);
	// generate training data
	x = (float**)calloc(n_samples, sizeof(float*));
	y = (float**)calloc(n_samples, sizeof(float*));
	mask = (1<<max_bit) - 1;
	for (i = 0; i < n_samples; ++i) {
		int a = kann_rand() & (mask>>1);
		int b = kann_rand() & (mask>>1);
		int c = (a + b) & mask; // NB: XOR is easier to learn than addition
		x[i] = (float*)calloc(max_bit * 2, sizeof(float));
		y[i] = (float*)calloc(max_bit * 2, sizeof(float));
		for (k = 0; k < max_bit; ++k) {
			x[i][k*2]   = (float)(a>>k&1);
			x[i][k*2+1] = (float)(b>>k&1);
			y[i][k*2 + (c>>k&1)] = 1.0f; // 1-hot encoding
		}
	}
	// train
	kann_train_fnn1(ann, 0.01f, 64, 25, 10, 0.1f, n_samples, x, y);
	// predict
	for (i = 0, n_err = 0; i < n_samples; ++i) {
		const float *y1 = kann_apply1(ann, x[i]);
		for (k = 0; k < max_bit; ++k)
			if ((y1[k*2] < y1[k*2+1]) != (y[i][k*2] < y[i][k*2+1]))
				++n_err;
	}
	fprintf(stderr, "Error rate per bit: %.2f%%\n", 100.0 * n_err / n_samples / max_bit);
	return 0;
}
```

[mlp]: https://en.wikipedia.org/wiki/Multilayer_perceptron
[cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
[lstm]: https://en.wikipedia.org/wiki/Long_short-term_memory
[ad]: https://en.wikipedia.org/wiki/Automatic_differentiation
[dh]: https://en.wikipedia.org/wiki/Dependency_hell
