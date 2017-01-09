This directory contains examples on using KANN as a library.

* [mlp.c](mlp.c): generic multi-layer perceptron. It reads TAB delimited files,
  where on each line, the first column is an arbirary name and the rest of
  columns gives a vector. On training, you need to provide two files, one for
  network input and one for output. On prediction, only network input is needed.
  ```sh
  curl -s https://url/to/mnist-data.tar | tar xf -
  ./mlp -o mnist-mlp.kan mnist-train-?.snd.gz
  ./mlp -i mnist-mlp.kan mnist-test-x.snd.gz | ./mnist-eval.pl
  ```

* [ae.c](ae.c): tied-weight denoising autoencoder. It takes the same format as
  `mlp.c`. This example shows how to construct a neural network with shared
  weights.

* [mnist-cnn.c](mnist-cnn.c): ConvNet for MNIST data.
  ```sh
  curl -s https://url/to/mnist-data.tar | tar xf -
  ./mnist-cnn -o mnist-cnn.kan mnist-train-?.snd.gz   # this will take a while
  ./mnist-cnn -i mnist-cnn.kan mnist-test-x.snd.gz | ./mnist-eval.pl
  ```

* [rnn-bit.c](rnn-bit.c): learn simple arithmetic (e.g. addition)
  ```sh
  seq 30000 | awk -v m=10000 '{a=int(m*rand());b=int(m*rand());print a,b,a+b}' \
    | ./rnn-bit -m5 -o add.kan -
  echo 400958 737471 | ./rnn-bit -Ai add.kan -
  ```

* [textgen.c](textgen.c): character-level text generation
  ```sh
  curl -s https://url/to/accelerando.txt.gz | gzip -dc > accelerando.txt
  ./textgen -o acc.kan accelerando.txt   # use -l2 -n256 for a better but slower model
  ./textgen -i acc.kan
  ```
