#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <stdio.h>
#include "kann.h"
#include "kann_data.h"

int main_mlp_train(int argc, char *argv[])
{
	int c, n_hidden_neurons = 50, n_hidden_layers = 1, seed = 11;
	kann_data_t *in = 0, *out = 0;
	kann_mopt_t mo;
	kann_t *ann;
	char *out_fn = 0, *in_fn = 0;

	kann_mopt_init(&mo);
	while ((c = getopt(argc, argv, "h:l:s:e:n:B:o:i:")) >= 0) {
		if (c == 'h') n_hidden_neurons = atoi(optarg);
		else if (c == 'l') n_hidden_layers = atoi(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 'e') mo.lr = atof(optarg);
		else if (c == 'n') mo.max_epoch = atoi(optarg);
		else if (c == 'B') mo.max_mbs = atoi(optarg);
		else if (c == 'i') in_fn = optarg;
		else if (c == 'o') out_fn = optarg;
	}
	if (argc - optind < 2) {
		fprintf(stderr, "Usage: kann mlp-train [options] <in.knd> <out.knd>\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  Model construction:\n");
		fprintf(stderr, "    -l INT      number of hidden layers [%d]\n", n_hidden_layers);
		fprintf(stderr, "    -h INT      number of hidden neurons per layer [%d]\n", n_hidden_neurons);
		fprintf(stderr, "    -s INT      random seed [%d]\n", seed);
		fprintf(stderr, "    -i FILE     read trained model from FILE []\n");
		fprintf(stderr, "    -o FILE     save trained model to FILE [stdout]\n");
		fprintf(stderr, "  Model training:\n");
		fprintf(stderr, "    -e FLOAT    learning rate [%g]\n", mo.lr);
		fprintf(stderr, "    -n INT      max number of epochs [%d]\n", mo.max_epoch);
		fprintf(stderr, "    -B INT      mini-batch size [%d]\n", mo.max_mbs);
		return 1;
	}

	in = kann_data_read(argv[optind]);
	out = kann_data_read(argv[optind+1]);
	assert(in->n_row == out->n_row);

	if (in_fn) {
		ann = kann_read(in_fn);
		assert(kann_n_in(ann) == in->n_col && kann_n_out(ann) == out->n_col);
	} else ann = kann_fnn_gen_mlp(in->n_col, out->n_col, n_hidden_layers, n_hidden_neurons, seed);

	kann_fnn_train(&mo, ann, in->n_row, in->x, out->x);
	if (out_fn) kann_write(out_fn, ann);

	kann_delete(ann);
	kann_data_free(out);
	kann_data_free(in);
	return 0;
}

int main_mlp_apply(int argc, char *argv[])
{
	kann_t *ann;
	kann_data_t *in;
	int c, i, j, n_out;

	while ((c = getopt(argc, argv, "")) >= 0) {
	}
	if (argc - optind < 2) {
		fprintf(stderr, "Usage: kann mlp-apply <model.knm> <in.knd>\n");
		return 1;
	}

	in = kann_data_read(argv[optind+1]);
	ann = kann_read(argv[optind]);
	assert(kann_n_in(ann) == in->n_col);
	n_out = kann_n_out(ann);
	for (i = 0; i < in->n_row; ++i) {
		const float *y;
		y = kann_fnn_apply1(ann, in->x[i]);
		if (in->rname) printf("%s\t", in->rname[i]);
		for (j = 0; j < n_out; ++j) {
			if (j) putchar('\t');
			printf("%.3g", y[j] + 1.0f - 1.0f);
		}
		putchar('\n');
	}
	kann_delete(ann);
	kann_data_free(in);
	return 0;
}

int main_rnn_train(int argc, char *argv[])
{
	int c, n_hidden_neurons = 50, n_hidden_layers = 1, seed = 11;
	kann_data_t *in = 0, *out = 0;
	kann_mopt_t mo;
	kann_t *ann;
	char *out_fn = 0, *in_fn = 0;

	kann_mopt_init(&mo);
	while ((c = getopt(argc, argv, "h:l:s:e:n:B:o:i:")) >= 0) {
		if (c == 'h') n_hidden_neurons = atoi(optarg);
		else if (c == 'l') n_hidden_layers = atoi(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 'e') mo.lr = atof(optarg);
		else if (c == 'n') mo.max_epoch = atoi(optarg);
		else if (c == 'B') mo.max_mbs = atoi(optarg);
		else if (c == 'i') in_fn = optarg;
		else if (c == 'o') out_fn = optarg;
	}
	if (argc - optind < 2) {
		fprintf(stderr, "Usage: kann rnn-train [options] <in.knd> <out.knd>\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  Model construction:\n");
		fprintf(stderr, "    -l INT      number of hidden layers [%d]\n", n_hidden_layers);
		fprintf(stderr, "    -h INT      number of hidden neurons per layer [%d]\n", n_hidden_neurons);
		fprintf(stderr, "    -s INT      random seed [%d]\n", seed);
		fprintf(stderr, "    -i FILE     read trained model from FILE []\n");
		fprintf(stderr, "    -o FILE     save trained model to FILE [stdout]\n");
		fprintf(stderr, "  Model training:\n");
		fprintf(stderr, "    -e FLOAT    learning rate [%g]\n", mo.lr);
		fprintf(stderr, "    -n INT      max number of epochs [%d]\n", mo.max_epoch);
		fprintf(stderr, "    -B INT      mini-batch size [%d]\n", mo.max_mbs);
		return 1;
	}

	in = kann_data_read(argv[optind]);
	out = kann_data_read(argv[optind+1]);
	assert(in->n_row == out->n_row);
//	ann = kann_rnn_gen_vanilla(in->n_col, out->n_col, n_hidden_layers, n_hidden_neurons, seed);
	ann = kann_rnn_gen_gru(in->n_col, out->n_col, n_hidden_layers, n_hidden_neurons, seed);
	/*
	if (in_fn) {
		ann = kann_read(in_fn);
		assert(kann_n_in(ann) == in->n_col && kann_n_out(ann) == out->n_col);
	} else ann = kann_gen_mlp(in->n_col, out->n_col, n_hidden_layers, n_hidden_neurons, seed);
	kann_train_fnn(&mo, ann, in->n_row, in->x, out->x);
	if (out_fn) kann_write(out_fn, ann);
	*/

	kann_delete(ann);
	kann_data_free(out);
	kann_data_free(in);
	return 0;
}
