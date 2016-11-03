#include <stdlib.h>
#include <string.h>
#include "kann_rand.h"
#include "kann.h"

kann_t *kann_mlp_gen(int n_in, int n_out, int n_hidden_layers, int n_hidden_neurons, uint64_t seed)
{
	int i, j, n_layers, n_par, *n_neurons;
	kad_node_t *in, *root, *out, *prev;
	kann_t *a;

	a = kann_init(seed);
	n_layers = n_hidden_layers + 2;
	n_neurons = (int*)alloca(n_layers * sizeof(int));
	n_neurons[0] = n_in, n_neurons[n_layers-1] = n_out;
	for (i = 1; i < n_layers - 1; ++i) n_neurons[i] = n_hidden_neurons;
	for (i = 1, n_par = 0; i < n_layers; ++i)
		n_par += n_neurons[i] * (n_neurons[i-1] + 1);
	a->t = (float*)calloc(n_par, sizeof(float));
	a->g = (float*)calloc(n_par, sizeof(float));

	prev = in = kad_par(0, 2, 1, n_in);
	in->label = KAD_LABEL_IN;
	out = kad_par(0, 2, 1, n_out);
	out->label = KAD_LABEL_OUT_TRUTH;
	for (i = 1, j = 0; i < n_layers; ++i) {
		kad_node_t *w, *b;
		if (i > 1) prev = kad_relu(prev);
		w = kad_var(&a->t[j], &a->g[j], 2, n_neurons[i], n_neurons[i-1]);
		kann_rand_weight(a->rng.data, n_neurons[i], n_neurons[i-1], &a->t[j]);
		j += n_neurons[i] * n_neurons[i-1];
		b = kad_var(&a->t[j], &a->g[j], 1, n_neurons[i]);
		j += n_neurons[i];
		prev = kad_add(kad_cmul(prev, w), b);
	}
	prev->label = KAD_LABEL_OUT_PRE;
	a->out_est = kad_sigm(prev);
	root = kad_ce2(prev, out);
	a->v = kad_compile(root, &a->n);
	kann_sync(a);
	return a;
}
