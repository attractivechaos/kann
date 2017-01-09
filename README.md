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

We developed KANN, 1) to understand the algorithms behind mainstream
frameworks; 2) to have a foundation flexible enough to experiment our own
contrived models; 3) to give other C/C++ programmers a tiny and efficient
library that can be easily integrated into their tools with no extra
dependencies. KANN is targeting small to medium neural networks that can be
trained on CPUs.

### Features

* Flexible. Model construction by building a computational graph with
  operators. Support RNNs, weight sharing and multiple inputs/outputs.

* Reasonably efficient. Support mini-batching. Optimized matrix product and
  convolution, coming close to (though not as fast as) OpenBLAS and mainstream
  deep learning frameworks. KANN may optionally work with BLAS libraries,
  enabled with the `HAVE_CBLAS` macro.

* Small. As of now, KANN has less than 3000 coding lines in four source code
  files, with no non-standard dependencies by default. We encourage developers
  to put all the KANN source code into their source code trees.

### Limitations

* CPU only; no parallelization. KANN does not support GPU or multithreading for
  now. As such, KANN is **not** intended for huge neural networks.

* Verbose APIs for training RNNs.

## Documentations

Comments in the header files briefly explain the APIs. More documentations can
be found in the [doc](doc) directory. Examples using the library can be found
in the [examples](examples) directory. The following is a complete program that
learns addition between two 10-bit integers (NB: RNN is a better and more
general model to learn additions; see also
[examples/rnn-bit.c](examples/rnn-bit.c)). This may help to give a
look-and-feel of KANN APIs on constructing and training simple feedforward
models.
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
	t = kann_layer_input(max_bit * 2);
	t = kad_relu(kann_layer_linear(t, 64));
	t = kad_relu(kann_layer_linear(t, 64));
	ann = kann_new(kann_layer_cost(t, max_bit * 2, KANN_C_CEB), 0);
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
