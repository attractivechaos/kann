#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include "kann.h"
#include "kann_data.h"

int main_mlp_train(int argc, char *argv[])
{
	int c, n_hidden_neurons = 50, n_hidden_layers = 1, seed = 11;
	kann_data_t *x = 0, *y = 0;
	kann_mopt_t mo;

	kann_mopt_init(&mo);
	while ((c = getopt(argc, argv, "h:l:s:e:")) != 0) {
		if (c == 'h') n_hidden_neurons = atoi(optarg);
		else if (c == 'l') n_hidden_layers = atoi(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 'e') mo.lr = atof(optarg);
	}
	if (argc - optind < 2) {
		fprintf(stderr, "Usage: kann mlp-train [options] <in.knd> <out.knd>\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  Model construction:\n");
		fprintf(stderr, "    -l INT      number of hidden layers [%d]\n", n_hidden_layers);
		fprintf(stderr, "    -h INT      number of hidden neurons per layer [%d]\n", n_hidden_neurons);
		fprintf(stderr, "    -s INT      random seed [%d]\n", seed);
		fprintf(stderr, "  Model training:\n");
		fprintf(stderr, "    -e FLOAT    learning rate [%g]\n", mo.lr);
		return 1;
	}
	return 0;
}
