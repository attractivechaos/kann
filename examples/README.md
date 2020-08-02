## Examples Using KANN

Data used for these examples and pre-trained KANN models from the data can be
downloaded from [this link][data]:
```sh
curl -LJs https://github.com/attractivechaos/kann/releases/download/v0/kann-data.tgz | tar -zxf -
curl -LJs https://github.com/attractivechaos/kann/releases/download/v0/kann-models.tgz | tar -zxf -
```

### Multi-layer perceptron

Implemented in [mlp.c](mlp.c). It reads TAB delimited files, where on each
line, the first column is an arbirary name and the rest of columns gives a
vector. On training, you need to provide two files, one for network input and
one for output. On prediction, only network input is needed.
```sh
./mlp -o mnist-mlp.kan kann-data/mnist-train-?.knd.gz
./mlp -i mnist-mlp.kan kann-data/mnist-test-x.knd.gz | kann-data/mnist-eval.pl
```

### Tied-weight denoising encoder

Implemented in [ae.c](ae.c). It takes the same format as `mlp.c`. This example
shows how to construct a neural network with shared weights.

### Variantional autoencoder

Implemented in [vae.c](vae.c). It uses sampling and a complex cost function.
```sh
./vae -o mnist-vae.kan -c 3 kann-data/mnist-train-x.knd.gz   # code dimension is 3
./vae -i mnist-vae.kan -A kann-data/mnist-test-x.knd.gz | kann-data/mnist-ascii.pl # reconstruction
./vae -i mnist-vae.kan -g 10 | kann-data/mnist-ascii.pl    # generate 10 random images
```

### CNN for MNIST

Implemented in [mnist-cnn.c](mnist-cnn.c).
```sh
./mnist-cnn -o mnist-cnn.kan -t4 kann-data/mnist-train-?.knd.gz
./mnist-cnn -i mnist-cnn.kan kann-data/mnist-test-x.knd.gz | kann-data/mnist-eval.pl
```

### RNN for simple arithmetic

Implemented in [rnn-bit.c](rnn-bit.c). This example can easily learn addition:
```sh
seq 30000 | awk -v m=10000 '{a=int(m*rand());b=int(m*rand());print a,b,a+b}' \
  | ./rnn-bit -m5 -o add.kan -
echo 400958 737471 | ./rnn-bit -Ai add.kan -
```
Although the model is trained on numbers below 10000, it can be applied to
larger numbers. This example can also learn simple `a*b` where `b` is a number
no more than 100:
```sh
./rnn-bit -m50 -l2 -n160 -o mul100.kan -t4 kann-data/mul100.train.txt
echo 15315611231621249 78 | ./rnn-bit -Ai mul100.kan -
```
A pre-trained model can be found in kann-models. There is also a [Keras-based
implementation](keras/rnn-bit.py). It does not converge. That is possibly
because KANN is taking initial hidden values as variables, which potentially
makes the model easier to learn. KANN uses layer normalization and dropout by
default, but without these operations, training does not stray too far away
like the python version.

### Character-level text generation with RNN

Implemented in [textgen.c](textgen.c). This is not a standard model in that the
initial hidden states depend on the previous output. It tends to memorize text
better.
```sh
./textgen -o acc.kan accelerando.txt
./textgen -i acc.kan
```
You can also found a bigger model in kann-models. It can generate meaningful
text even with near-to-zero temperature.
```sh
./textgen -i kann-models/acc-l3-n256r.kan -T 1e-6
```

[data]: https://github.com/attractivechaos/kann/releases/tag/v0
