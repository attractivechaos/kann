.
|-- accelerando.txt         # Reformantted text for Accelerando (a free Science Fiction)
|-- mnist-ascii.pl          # Convert MNIST data to ASCII images
|-- mnist-eval.pl           # Evaluate the accuracy of MNIST
|-- mnist-test-x.knd.gz     # 10000 MNIST test images, formated in the KANN data format
|-- mnist-test-y.knd.gz     # 10000 MNIST test labels
|-- mnist-train-x.knd.gz    # 60000 MNIST training images
|-- mnist-train-y.knd.gz    # 60000 MNIST training labels
`-- mul100.train.txt        # 30000 x*y samples to train rnn-bit (0<=x<10000 && 1<=y<=100)

Example command lines:

	kann/example/textgen -o acc-simple.kan accelerando.txt
	kann/example/textgen -i acc-simple.kan

	kann/example/mlp -o mnist-mlp.kan mnist-train-x.knd.gz mnist-train-y.knd.gz
	kann/example/mlp -i mnist-mlp.kan mnist-test-x.knd.gz | ./mnist-eval.pl

	kann/example/mnist-cnn -o mnist-cnn.kan mnist-train-x.knd.gz mnist-train-y.knd.gz
	kann/example/mnist-cnn -i mnist-cnn.kan mnist-test-x.knd.gz | ./mnist-eval.pl

	kann/example/vae -o mnist-vae.kan mnist-train-x.knd.gz
	kann/example/vae -Ai mnist-vae.kan mnist-test-x.knd.gz | ./mnist-ascii.pl | less
	kann/example/vae -g 10 -i mnist-vae.kan | ./mnist-ascii.pl
