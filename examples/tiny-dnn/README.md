To compile examples in this directory:
```sh
wget -O- http://url-to/kann-data.tgz | tar zxf -
git clone https://github.com/tiny-dnn/tiny-dnn
make TINY_DNN=tiny-dnn
./mlp -o model.tdm kann-data/mnist-train-*
./mlp -i model.tdm kann-data/mnist-test-x.knd.gz | kann-data/mnist-eval.pl
```
By default, Makefile asks tiny-dnn to use AVX but restrict to one CPU core.
For small models, using multiple cores is actually slower.

On my laptop, tiny-dnn+AVX here is 6.7 times as slow as the same 1-layer MLP on
top of KANN+SSE.
