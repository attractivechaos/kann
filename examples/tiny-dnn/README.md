To compile examples in this directory:
```sh
wget -O- http://url-to/kann-data.tgz | tar zxf -
make TINY_DNN=/path-to/tiny-dnn
./mlp -o model.tdm kann-data/mnist-train-*
./mlp -i model.tdm kann-data/mnist-test-x.knd.gz | kann-data/mnist-eval.pl
```
