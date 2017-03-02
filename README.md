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
# learn multiplication to a number between 1 and 100 (with a bigger model)
seq 30000 | awk '{a=int(10000*rand());b=int(100*rand())+1;print a,b,a*b}' \
  | ./examples/rnn-bit -m50 -l2 -n160 -t4 -o mul100.kan -
# apply the model to large numbers (answer: 1194617676066457422)
echo 15315611231621249 78 | ./examples/rnn-bit -Ai mul100.kan -
```

## Introduction

KANN is a standalone and lightweight library in C for constructing and training
small to medium artificial neural networks such as [multi-layer
perceptrons][mlp], [convolutional neural networks][cnn] and [recurrent neural
networks][rnn] (including [LSTM][lstm] and [GRU][gru]). It implements
graph-based reverse-mode [automatic differentiation][ad] and allows to build
topologically complex neural networks with recurrence, shared weights and
multiple inputs/outputs/costs. In comparison to mainstream deep learning
frameworks such as [TensorFlow][tf], KANN is not as scalable, but it is close
in flexibility, has a much smaller code base and only depends on the standard C
library. In comparison to other lightweight frameworks such as [tiny-dnn][td],
KANN is still smaller, times faster and much more versatile, supporting RNN,
VAE and non-standard neural networks that may fail these lightweight
frameworks.

KANN could be potentially useful when you want to experiment small to medium
neural networks in C/C++, to deploy no-so-large models without worrying about
[dependency hell][dh], or to learn the internals of deep learning libraries.

### Features

* Flexible. Model construction by building a computational graph with
  operators. Support RNNs, weight sharing and multiple inputs/outputs.

* Efficient. Reasonably optimized matrix product and convolution. Support
  mini-batching and effective multi-threading. Sometimes faster than mainstream
  frameworks in their CPU-only mode.

* Small. As of now, KANN has less than 4000 lines of code in four source code
  files, with no non-standard dependencies by default.

### Limitations

* CPU only. As such, KANN is **not** intended for training huge neural
  networks.

* Bidirectional RNNs and seq2seq models require manual unrolling, which is
  complex and tedious. No batch normalization.

* Verbose APIs for training RNNs.

## Installation

The KANN library is composed of four files: `kautodiff.{h,c}` and `kann.{h,c}`.
You are encouraged to include these files in your source code tree. No
installation is needed. To compile examples:
```sh
make
```
This generates a few executables in the [examples](examples) directory.

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

This example learns to count the number of "1" bits in an integer (i.e.
popcount):
```c
// to compile and run: gcc -O2 this-prog.c kann.c kautodiff.c && ./a.out
#include <stdlib.h>
#include <stdio.h>
#include "kann.h"

int main(void)
{
	int i, k, max_bit = 20, n_samples = 30000, mask = (1<<max_bit)-1, n_err, max_k;
	float **x, **y, max, *x1;
	kad_node_t *t;
	kann_t *ann;
	// construct an MLP with one hidden layers
	t = kann_layer_input(max_bit);
	t = kad_relu(kann_layer_linear(t, 64));
	t = kann_layer_cost(t, max_bit, KANN_C_CEM); // output uses 1-hot encoding
	ann = kann_new(t, 0);
	// generate training data
	x = (float**)calloc(n_samples, sizeof(float*));
	y = (float**)calloc(n_samples, sizeof(float*));
	for (i = 0; i < n_samples; ++i) {
		int c, a = kad_rand(0) & (mask>>1);
		x[i] = (float*)calloc(max_bit, sizeof(float));
		y[i] = (float*)calloc(max_bit, sizeof(float));
		for (k = c = 0; k < max_bit; ++k)
			x[i][k] = (float)(a>>k&1), c += (a>>k&1);
		y[i][c] = 1.0f;
	}
	// train
	kann_train_fnn1(ann, 0.001f, 64, 50, 10, 0.1f, n_samples, x, y);
	// predict
	x1 = (float*)calloc(max_bit, sizeof(float));
	for (i = n_err = 0; i < n_samples; ++i) {
		int c, a = kad_rand(0) & (mask>>1); // generating a new number
		const float *y1;
		for (k = c = 0; k < max_bit; ++k)
			x1[k] = (float)(a>>k&1), c += (a>>k&1);
		y1 = kann_apply1(ann, x1);
		for (k = 0, max_k = -1, max = -1.0f; k < max_bit; ++k) // find the max
			if (max < y1[k]) max = y1[k], max_k = k;
		if (max_k != c) ++n_err;
	}
	fprintf(stderr, "Test error rate: %.2f%%\n", 100.0 * n_err / n_samples);
	kann_delete(ann); // TODO: also to free x, y and x1
	return 0;
}
```

## Benchmarks

|Task       |Framework   |Machine|Device   |Real time|CPU time|Command line |
|:----------|:-----------|:------|--------:|--------:|-------:|:------------|
|MNIST-mlp  |KANN+SSE    |Linux  |1 CPU    | 31.3s   | 31.2s  |mlp -m20 -v0|
|           |            |Mac    |1 CPU    | 27.1s   | 27.1s  |mlp -m20 -v0|
|           |Theano+Keras|Linux  |1 CPU    | 33.7s   | 33.2s  |OMP_NUM_THREADS=1 keras/mlp.py -m20 -v0|
|           |            |       |4 CPUs   | 32.0s   |121.3s  |OMP_NUM_THREADS=4 ... |
|           |            |Mac    |1 CPU    | 37.2s   | 35.2s  ||
|           |            |       |2 CPUs   | 32.9s   | 62.0s  ||
|           |TF+Keras    |Mac    |1-2 CPUs | 64.2s   |102.0s  |keras/mlp.py -m20|
|           |TensorFlow  |Mac    |1 CPU    | 33.4s   | 33.4s  |tensorflow/mlp.py -m20|
|           |            |       |2 CPUs   | 29.2s   | 50.6s  |tensorflow/mlp.py -m20 -t2|
|           |Tiny-dnn    |Linux  |1 CPU    | 2m19s   | 2m18s  |tiny-dnn/mlp -m20|
|           |Tiny-dnn+AVX|Linux  |1 CPU    | 1m34s   | 1m33s  |tiny-dnn/mlp -m20|
|           |            |Mac    |1 CPU    | 2m17s   | 2m16s  |tiny-dnn/mlp -m20|
|MNIST-cnn  |KANN+SSE    |Linux  |1 CPU    |57m57s   |57m53s  |mnist-cnn -v0 -t4 -m15|
|           |            |       |4 CPUs   |19m09s   |68m17s  |mnist-cnn -v0 -t4 -m15|
|           |Theano+Keras|Linux  |1 CPU    |37m12s   |37m09s  |OMP_NUM_THREADS=1 keras/mlp.py -Cm15 -v0|
|           |            |       |4 CPUs   |24m24s   |97m22s  |OMP_NUM_THREADS=4 keras/mlp.py -Cm15 -v0|
|           |Tiny-dnn+AVX|Linux  |1 CPU    |         |        |tiny-dnn/mlp -Cm15|
|mul100-rnn |KANN+SSE    |Linux  |1 CPU    |40m05s   |40m02s  |rnn-bit -l2 -n160 -m25 -Nd0|
|           |            |       |4 CPUs   |12m13s   |44m40s  |rnn-bit -l2 -n160 -t4 -m25 -Nd0|
|           |Theano+Keras|Linux  |1 CPU    |27m30s   |27m27s  |OMP_NUM_THREADS=1 keras/rnn-bit -l2 -n160 -m25|
|           |            |       |4 CPUs   |19m52m   |77m45s  |OMP_NUM_THREADS=4 keras/rnn-bit -l2 -n160 -m25|



[mlp]: https://en.wikipedia.org/wiki/Multilayer_perceptron
[cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
[rnn]: https://en.wikipedia.org/wiki/Recurrent_neural_network
[gru]: https://en.wikipedia.org/wiki/Gated_recurrent_unit
[lstm]: https://en.wikipedia.org/wiki/Long_short-term_memory
[ad]: https://en.wikipedia.org/wiki/Automatic_differentiation
[dh]: https://en.wikipedia.org/wiki/Dependency_hell
[ae]: https://en.wikipedia.org/wiki/Autoencoder
[vae]: https://en.wikipedia.org/wiki/Autoencoder#Variational_autoencoder_.28VAE.29
[tf]: https://www.tensorflow.org
[td]: https://github.com/tiny-dnn/tiny-dnn
