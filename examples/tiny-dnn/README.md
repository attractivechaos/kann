To compile examples in this directory:
```sh
wget -O- http://url-to/kann-data.tgz | tar zxf -
git clone https://github.com/tiny-dnn/tiny-dnn
make TINY_DNN=tiny-dnn
./mlp -o model.tdm kann-data/mnist-train-*
./mlp -i model.tdm kann-data/mnist-test-x.knd.gz | kann-data/mnist-eval.pl
```

On my laptop, mlp here is 9.6-fold as slow as the same 1-layer MLP on top of
KANN. Tiny-dnn is slower mainly because its default matrix multiplication
routine causes excessive cache misses.
