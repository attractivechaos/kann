#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include "tiny_dnn/tiny_dnn.h"
#include "kann_extra/kann_data.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

network<sequential> mlp_model_gen(int n_in, int n_out, int n_layer, int n_hidden)
{
	network<sequential> nn;
	int n_last = n_in;
	for (int i = 0; i < n_layer; ++i) {
		nn << fully_connected_layer<relu>(n_last, n_hidden);
		n_last = n_hidden;
	}
	nn << fully_connected_layer<softmax>(n_last, n_out);
	return nn;
}

network<sequential> mnist_cnn_model_gen(void)
{
	network<sequential> nn;
	nn << convolutional_layer<relu>(28, 28, 3, 1, 32)
	   << convolutional_layer<relu>(26, 26, 3, 32, 32)
	   << max_pooling_layer<identity>(24, 24, 32, 2)
	   << fully_connected_layer<relu>(12 * 12 * 32, 128)
	   << fully_connected_layer<softmax>(128, 10);
	return nn;
}

void mlp_float2vec(std::vector<vec_t> &data, int n, int m, float **x)
{
	for (int i = 0; i < n; ++i) {
		vec_t d;
		for (int j = 0; j < m; ++j) d.push_back(x[i][j]);
		data.push_back(d);
	}
}

int main(int argc, char *argv[])
{
	int c, n_layer = 1, n_hidden = 64, minibatch = 64, max_epoch = 20, mnist_cnn = 0;
	kann_data_t *kdx;
	float lr = 0.001f;
	char *fn_out = 0, *fn_in = 0;

	while ((c = getopt(argc, argv, "i:o:m:B:l:n:r:C")) >= 0) {
		if (c == 'o') fn_out = optarg;
		else if (c == 'i') fn_in = optarg;
		else if (c == 'm') max_epoch = atoi(optarg);
		else if (c == 'B') minibatch = atoi(optarg);
		else if (c == 'l') n_layer = atoi(optarg);
		else if (c == 'n') n_hidden = atoi(optarg);
		else if (c == 'r') lr = atof(optarg);
		else if (c == 'C') mnist_cnn = 1;
	}
	if (argc - optind < 1) {
		fprintf(stderr, "Usage: mlp [options] <in.knd> [out.knd]\n");
		return 1;
	}

	kdx = kann_data_read(argv[optind]);
	if (argc - optind >= 2) { // training
		std::vector<vec_t> dx, dy;
		kann_data_t *kdy = kann_data_read(argv[optind+1]);
		int n = kdx->n_row, n_in = kdx->n_col, n_out = kdy->n_col;

		auto nn = mnist_cnn? mnist_cnn_model_gen() : mlp_model_gen(n_in, n_out, n_layer, n_hidden);
		mlp_float2vec(dx, n, n_in, kdx->x);
		mlp_float2vec(dy, n, n_out, kdy->x);

		gradient_descent optimizer;
		optimizer.alpha = lr * minibatch;

		progress_display disp(static_cast<unsigned long>(n));
		auto on_enumerate_epoch = [&]() { disp.restart(static_cast<unsigned long>(n)); };
		auto on_enumerate_minibatch = [&]() { disp += minibatch; };

		nn.fit<cross_entropy_multiclass>(optimizer, dx, dy, minibatch, max_epoch, on_enumerate_minibatch, on_enumerate_epoch);
		if (fn_out) nn.save(fn_out);

		kann_data_free(kdy);
	} else if (fn_in) {
		network<sequential> nn;
		std::vector<vec_t> dx;
		mlp_float2vec(dx, kdx->n_row, kdx->n_col, kdx->x);
		nn.load(fn_in);
		for (int i = 0; i < kdx->n_row; ++i) {
			vec_t y = nn.predict(dx[i]);
			printf("%s", kdx->rname[i]);
			for (int j = 0; j < y.size(); ++j)
				printf("\t%g", y[j] + 1.0f - 1.0f);
			putchar('\n');
		}
	}
	kann_data_free(kdx);
	return 0;
}
