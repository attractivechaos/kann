#include <stdlib.h>
#include <string.h>
#include "kann_rand.h"
#include "kann.h"

kad_node_t *kann_new_weight(int n_row, int n_col, void *rng_data)
{
	kad_node_t *w;
	w = kad_var(0, 0, 2, n_row, n_col);
	w->x = (float*)malloc(n_row * n_col * sizeof(float));
	kann_rand_weight(rng_data, n_row, n_col, w->x);
	return w;
}

kad_node_t *kann_new_bias(int n, void *rng_data)
{
	kad_node_t *b;
	b = kad_var(0, 0, 1, n);
	b->x = (float*)calloc(n, sizeof(float));
	return b;
}

kann_t *kann_fnn_gen_mlp(int n_in, int n_out, int n_hidden_layers, int n_hidden_neurons, uint64_t seed)
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
		w = kann_new_weight(n_neurons[i], n_neurons[i-1], a->rng.data);
		b = kann_new_bias(n_neurons[i], a->rng.data);
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

kann_t *kann_rnn_gen_vanilla(int n_in, int n_out, int n_hidden_layers, int n_hidden_neurons, uint64_t seed)
{
	int i, n_layers, *n_neurons;
	kad_node_t **h_in, **h_out;
	kann_t *a;

	a = kann_init(seed);
	n_layers = n_hidden_layers + 2;
	n_neurons = (int*)alloca(n_layers * sizeof(int));
	n_neurons[0] = n_in, n_neurons[n_layers-1] = n_out;
	for (i = 1; i < n_layers - 1; ++i) n_neurons[i] = n_hidden_neurons;
	h_in = (kad_node_t**)alloca(n_layers * sizeof(kad_node_t*));
	h_out = (kad_node_t**)alloca(n_layers * sizeof(kad_node_t*));

	h_out[0] = kad_par(0, 2, 1, n_in);
	for (i = 1; i < n_layers - 1; ++i)
		h_in[i] = kad_par(0, 2, 1, n_hidden_neurons);
	for (i = 1; i < n_layers; ++i) {
		kad_node_t *w, *u, *b;
		w = kann_new_weight(n_neurons[i], n_neurons[i-1], a->rng.data);
		b = kann_new_bias(n_neurons[i], a->rng.data);
		u = kann_new_weight(n_neurons[i], n_neurons[i], a->rng.data);
		if (i < n_layers - 1) {
			h_out[i] = kad_relu(kad_add(kad_add(kad_cmul(h_out[i-1], w), kad_cmul(h_in[i], u)), b));
			h_out[i]->pre = h_in[i];
		} else h_out[i] = kad_sigm(kad_add(kad_cmul(h_out[i-1], w), b));
	}
	h_out[0]->label = KANN_LABEL_IN;
	h_out[n_layers - 2]->label = KANN_LABEL_LAST;
	h_out[n_layers - 1]->label = KANN_LABEL_OUT;
	a->v = kad_compile(&a->n, 1, h_out[n_layers - 1]);
	kann_collate_var(a);

#if 0
//	kad_debug(stderr, a->n, a->v);
	int n_u;
	kad_node_t **u;
	u = kad_unroll(a->n, a->v, 3, &n_u);
	kad_debug(stderr, n_u, u);
#endif
	return a;
}

kann_t *kann_gen_gru(int n_in, int n_out, int n_hidden_layers, int n_hidden_neurons, uint64_t seed)
{
	return 0;
}

kann_t *kann_gen_lstm(int n_in, int n_out, int n_hidden_layers, int n_hidden_neurons, uint64_t seed)
{
	return 0;
}
