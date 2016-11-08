#include <stdlib.h>
#include <string.h>
#include "kann_rand.h"
#include "kann.h"

kann_t *kann_mlp_gen(int n_in, int n_out, int n_hidden_layers, int n_hidden_neurons, uint64_t seed)
{
	int i, n_layers, *n_neurons;
	kad_node_t *in, *out, *truth, *prev, *cost;
	kann_t *a;

	a = kann_init(seed);
	n_layers = n_hidden_layers + 2;
	n_neurons = (int*)alloca(n_layers * sizeof(int));
	n_neurons[0] = n_in, n_neurons[n_layers-1] = n_out;
	for (i = 1; i < n_layers - 1; ++i) n_neurons[i] = n_hidden_neurons;

	prev = in = kad_par(0, 2, 1, n_in);
	truth = kad_par(0, 2, 1, n_out);
	for (i = 1; i < n_layers; ++i) {
		kad_node_t *w, *b;
		if (i > 1) prev = kad_relu(prev);
		w = kad_var(0, 0, 2, n_neurons[i], n_neurons[i-1]);
		w->x = (float*)malloc(n_neurons[i] * n_neurons[i-1] * sizeof(float));
		kann_rand_weight(a->rng.data, n_neurons[i], n_neurons[i-1], w->x);
		b = kad_var(0, 0, 1, n_neurons[i]);
		b->x = (float*)calloc(n_neurons[i], sizeof(float));
		prev = kad_add(kad_cmul(prev, w), b);
	}
	out = kad_sigm(prev);
	cost = kad_ce2(prev, truth);
	in->label = KANN_LABEL_IN;
	truth->label = KANN_LABEL_TRUTH;
	out->label = KANN_LABEL_OUT;
	cost->label = KANN_LABEL_COST;
	a->v = kad_compile(&a->n, 2, out, cost);
	kann_collate_var(a);
	kann_sync_index(a);
	return a;
}
