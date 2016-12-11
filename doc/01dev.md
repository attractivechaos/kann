### N-dimensional array

N-dimensional array, or n-d array in brief, is a fundamental object that holds
most types of numerical data in KANN. It can be described with the following
struct:
```cpp
typedef struct {
    int n_d;   // number of dimensions
    int *d;    // dimensions, of size n_d
    float *x;  // array data, of size \prod_i{d[i]} (1 if n_d==0)
} NDArray_t;
```
The dimensions are also called the *shape* of the array. Deep learning
frameworks often take n-d array as a synonym of *tensor*, though according to
[wiki][tensor-wiki], this seems imprecise. Conventionally, we call the n-d
array a scalar if *n\_d* equals 0, a vector if *n\_d* equals 1 and a matrix if
*n\_d* equals 2.

### Automatic differentiation and computational graph

[Automatic differentiation][ad] is the backbone of several major deep learning
frameworks such as [TensorFlow][tf] and [Theano][theano]. It efficiently
computes the gradient of a function without symbolic derivation. Automatic
differentiation is typically achieved with a graph which is sometimes called as
a computational graph or a data flow graph (as in TensorFlow). Detailed
description of automatic differentiation is beyond the scope of this note. We
will only show an example here:

![](autodiff.png)

Files [kautodiff.*](../kautodiff.h) implement automatic differentiation.

### Network layers

Many deep learning libraries use the concept of *layer* to build a model.
In the context of computational graph, a layer is a well-defined reusable
subgraph. Such layers are more flexible as we easily allow non-sequential
combinations of subgraphs. The following shows a the computational graph of a
[multilayer perceptron][mlp]:

![](mlp.png)

In this figure, each dotted red box represents a dense (aka fully connected)
layer that has one input (in green) and one output (in blue). In KANN, a layer
works exactly this way. Layer subgraphs are defined functions `kann_layer_*()`
in [kann.*](../kann.h).


[tensor-wiki]: https://en.wikipedia.org/wiki/Tensor
[tf]: https://www.tensorflow.org
[theano]: http://deeplearning.net/software/theano/
[ad]: https://en.wikipedia.org/wiki/Automatic_differentiation
[mlp]: https://en.wikipedia.org/wiki/Multilayer_perceptron
