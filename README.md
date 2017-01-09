## Introduction

KANN is a standalone 4-file library in C for constructing and training
artificial neural networks such as [MLP][mlp], [CNN][cnn] and [LSTM][lstm]. It
implements generic reverse-mode [automatic differentiation][ad] based on the
concept of computational graph and allows to construct neural networks with
shared weights, multiple inputs/outputs and recurrence. KANN is portable, small
and fairly efficient for its size.

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

* CPU only. KANN does not support GPU, at least for now.

* Verbose APIs for training RNNs.

[mlp]: https://en.wikipedia.org/wiki/Multilayer_perceptron
[cnn]: https://en.wikipedia.org/wiki/Convolutional_neural_network
[lstm]: https://en.wikipedia.org/wiki/Long_short-term_memory
[ad]: https://en.wikipedia.org/wiki/Automatic_differentiation
