## Getting Started
```sh
# acquire source code and compile
git clone https://github.com/attractivechaos/kann
cd kann; make  # or "make CBLAS=/path/to/openblas" for faster matrix multiplication
# learn unsigned addition (30000 samples; numbers within 10000)
seq 30000 | awk -v m=10000 '{a=int(m*rand());b=int(m*rand());print a,b,a+b}' \
  | ./examples/rnn-bit -m7 -o add.kan -
# apply the model (output 1138429, the sum of the two numbers)
echo 400958 737471 | ./examples/rnn-bit -Ai add.kan -
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

* Small and portable. As of now, KANN has less than 4000 lines of code in four
  source code files, with no non-standard dependencies by default. Compatible with 
  ANSI C compilers.

### Limitations

* CPU only. As such, KANN is **not** intended for training huge neural
  networks.

* Lack of some common operators and architectures such as batch normalization.

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
t = kad_relu(kann_layer_dense(t, 64)); // a 64-neuron hidden layer with ReLU activation
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
// to compile and run: gcc -O2 this-prog.c kann.c kautodiff.c -lm && ./a.out
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
	t = kad_relu(kann_layer_dense(t, 64));
	t = kann_layer_cost(t, max_bit + 1, KANN_C_CEM); // output uses 1-hot encoding
	ann = kann_new(t, 0);
	// generate training data
	x = (float**)calloc(n_samples, sizeof(float*));
	y = (float**)calloc(n_samples, sizeof(float*));
	for (i = 0; i < n_samples; ++i) {
		int c, a = kad_rand(0) & (mask>>1);
		x[i] = (float*)calloc(max_bit, sizeof(float));
		y[i] = (float*)calloc(max_bit + 1, sizeof(float));
		for (k = c = 0; k < max_bit; ++k)
			x[i][k] = (float)(a>>k&1), c += (a>>k&1);
		y[i][c] = 1.0f; // c is ranged from 0 to max_bit inclusive
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
		for (k = 0, max_k = -1, max = -1.0f; k <= max_bit; ++k) // find the max
			if (max < y1[k]) max = y1[k], max_k = k;
		if (max_k != c) ++n_err;
	}
	fprintf(stderr, "Test error rate: %.2f%%\n", 100.0 * n_err / n_samples);
	kann_delete(ann); // TODO: also to free x, y and x1
	return 0;
}
```

## Benchmarks

* First of all, this benchmark only evaluates relatively small networks, but
  in practice, it is huge networks on GPUs that really demonstrate the true
  power of mainstream deep learning frameworks. *Please don't read too much into
  the table*.

* "Linux" has 48 cores on two Xeno E5-2697 CPUs at 2.7GHz. MKL, NumPy-1.12.0
  and Theano-0.8.2 were installed with Conda; Keras-1.2.2 installed with pip.
  The official TensorFlow-1.0.0 wheel does not work with Cent OS 6 on this
  machine, due to glibc. This machine has one Tesla K40c GPU installed. We are
  using by CUDA-7.0 and cuDNN-4.0 for training on GPU.

* "Mac" has 4 cores on a Core i7-3667U CPU at 2GHz. MKL, NumPy and Theano came
  with Conda, too. Keras-1.2.2 and Tensorflow-1.0.0 were installed with pip. On
  both machines, Tiny-DNN was acquired from github on March 1st, 2017.

* mnist-mlp implements a simple MLP with one layer of 64 hidden neurons.
  mnist-cnn applies two convolutional layers with 32 3-by-3 kernels and ReLU
  activation, followed by 2-by-2 max pooling and one 128-neuron dense layer.
  mul100-rnn uses two GRUs of size 160. Both input and output are 2-D
  binary arrays of shape (14,2) -- 28 GRU operations for each of the 30000
  training samples.

|Task       |Framework    |Machine|Device   |Real     |CPU     |Command line |
|:----------|:------------|:------|--------:|--------:|-------:|:------------|
|mnist-mlp  |KANN+SSE     |Linux  |1 CPU    | 31.3s   | 31.2s  |mlp -m20 -v0|
|           |             |Mac    |1 CPU    | 27.1s   | 27.1s  ||
|           |KANN+BLAS    |Linux  |1 CPU    | 18.8s   | 18.8s  ||
|           |Theano+Keras |Linux  |1 CPU    | 33.7s   | 33.2s  |keras/mlp.py -m20 -v0|
|           |             |       |4 CPUs   | 32.0s   |121.3s  ||
|           |             |Mac    |1 CPU    | 37.2s   | 35.2s  ||
|           |             |       |2 CPUs   | 32.9s   | 62.0s  ||
|           |TensorFlow   |Mac    |1 CPU    | 33.4s   | 33.4s  |tensorflow/mlp.py -m20|
|           |             |       |2 CPUs   | 29.2s   | 50.6s  |tensorflow/mlp.py -m20 -t2|
|           |Tiny-dnn     |Linux  |1 CPU    | 2m19s   | 2m18s  |tiny-dnn/mlp -m20|
|           |Tiny-dnn+AVX |Linux  |1 CPU    | 1m34s   | 1m33s  ||
|           |             |Mac    |1 CPU    | 2m17s   | 2m16s  ||
|mnist-cnn  |KANN+SSE     |Linux  |1 CPU    |57m57s   |57m53s  |mnist-cnn -v0 -m15|
|           |             |       |4 CPUs   |19m09s   |68m17s  |mnist-cnn -v0 -t4 -m15|
|           |Theano+Keras |Linux  |1 CPU    |37m12s   |37m09s  |keras/mlp.py -Cm15 -v0|
|           |             |       |4 CPUs   |24m24s   |97m22s  ||
|           |             |       |1 GPU    |2m57s    |        |keras/mlp.py -Cm15 -v0|
|           |Tiny-dnn+AVX |Linux  |1 CPU    |300m40s  |300m23s |tiny-dnn/mlp -Cm15|
|mul100-rnn |KANN+SSE     |Linux  |1 CPU    |40m05s   |40m02s  |rnn-bit -l2 -n160 -m25 -Nd0|
|           |             |       |4 CPUs   |12m13s   |44m40s  |rnn-bit -l2 -n160 -t4 -m25 -Nd0|
|           |KANN+BLAS    |Linux  |1 CPU    |22m58s   |22m56s  |rnn-bit -l2 -n160 -m25 -Nd0|
|           |             |       |4 CPUs   |8m18s    |31m26s  |rnn-bit -l2 -n160 -t4 -m25 -Nd0|
|           |Theano+Keras |Linux  |1 CPU    |27m30s   |27m27s  |rnn-bit.py -l2 -n160 -m25|
|           |             |       |4 CPUs   |19m52s   |77m45s  ||

* In the single thread mode, Theano is about 50% faster than KANN probably due
  to efficient matrix multiplication (aka. `sgemm`) implemented in MKL. As is
  shown in a [previous micro-benchmark][matmul], MKL/OpenBLAS can be twice as
  fast as the implementation in KANN.

* KANN can optionally use the `sgemm` routine from a BLAS library (enabled by
  macro `HAVE_CBLAS`). Linked against OpenBLAS-0.2.19, KANN matches the
  single-thread performance of Theano on Mul100-rnn. KANN doesn't reduce
  convolution to matrix multiplication, so MNIST-cnn won't benefit from
  OpenBLAS. We observed that OpenBLAS is slower than the native KANN
  implementation when we use a mini-batch of size 1. The cause is unknown.

* KANN's intra-batch multi-threading model is better than Theano+Keras.
  However, in its current form, this model probably won't get alone well with
  GPUs.



[mlp]: https://en.wikipedia.org/wiki/Multilayer_perceptron
[cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
[rnn]: https://en.wikipedia.org/wiki/Recurrent_neural_network
[gru]: https://en.wikipedia.org/wiki/Gated_recurrent_unit
[lstm]: https://en.wikipedia.org/wiki/Long_short-term_memory
[ad]: https://en.wikipedia.org/wiki/Automatic_differentiation
[dh]: https://en.wikipedia.org/wiki/Dependency_hell
[ae]: https://en.wikipedia.org/wiki/Autoencoder
[tf]: https://www.tensorflow.org
[td]: https://github.com/tiny-dnn/tiny-dnn
[matmul]: https://github.com/attractivechaos/matmul
