#include <stdlib.h>
#include <string.h>
#include "kann_rand.h"
#include "kann.h"

/*********************************
 * Weight matrix and bias vector *
 *********************************/

kad_node_t *kann_new_weight(int n_row, int n_col)
{
	kad_node_t *w;
	w = kad_var(0, 0, 2, n_row, n_col);
	w->x = (float*)malloc(n_row * n_col * sizeof(float));
	kann_rand_weight(n_row, n_col, w->x);
	return w;
}

kad_node_t *kann_new_bias(int n)
{
	kad_node_t *b;
	b = kad_var(0, 0, 1, n);
	b->x = (float*)calloc(n, sizeof(float));
	return b;
}

/*****************
 * Common layers *
 *****************/

kad_node_t *kann_layer_linear(kad_node_t *in, int n1)
{
	int n0;
	kad_node_t *w, *b;
	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	w = kann_new_weight(n1, n0);
	b = kann_new_bias(n1);
	return kad_add(kad_cmul(in, w), b);
}

kad_node_t *kann_layer_gru(kad_node_t *in, int n1)
{
	int n0;
	kad_node_t *r, *z, *w, *u, *b, *s, *h0, *out;

	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	h0 = kad_var(0, 0, 2, 1, n1);
	h0->x = (float*)calloc(n1, sizeof(float));

	// z = sigm(x_t * W_z + h_{t-1} * U_z + b_z)
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	z = kad_sigm(kad_add(kad_add(kad_cmul(in, w), kad_cmul(h0, u)), b));
	// r = sigm(x_t * W_r + h_{t-1} * U_r + b_r)
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	r = kad_sigm(kad_add(kad_add(kad_cmul(in, w), kad_cmul(h0, u)), b));
	// s = tanh(x_t * W_s + (h_{t-1} # r) * U_s + b_s)
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	s = kad_tanh(kad_add(kad_add(kad_cmul(in, w), kad_cmul(kad_mul(r, h0), u)), b));
	// h_t = z # h_{t-1} + (1 - z) # s
	out = kad_add(kad_mul(kad_1minus(z), s), kad_mul(z, h0));
	out->pre = h0;
	return out;
}

/*****************
 * Common models *
 *****************/

kann_t *kann_fnn_gen_mlp(int n_in, int n_out, int n_hidden_layers, int n_hidden_neurons)
{
	int i, n_layers, *n_neurons;
	kad_node_t *in, *out, *truth, *prev, *cost;
	kann_t *a;

	a = kann_new();
	n_layers = n_hidden_layers + 2;
	n_neurons = (int*)alloca(n_layers * sizeof(int));
	n_neurons[0] = n_in, n_neurons[n_layers-1] = n_out;
	for (i = 1; i < n_layers - 1; ++i) n_neurons[i] = n_hidden_neurons;

	prev = in = kad_par(0, 2, 1, n_in), in->label = KANN_L_IN;
	truth = kad_par(0, 2, 1, n_out), truth->label = KANN_L_TRUTH;
	for (i = 1; i < n_layers; ++i) {
		kad_node_t *w, *b;
		if (i > 1) prev = kad_relu(prev);
		w = kann_new_weight(n_neurons[i], n_neurons[i-1]);
		b = kann_new_bias(n_neurons[i]);
		prev = kad_add(kad_cmul(prev, w), b);
	}
	out = kad_sigm(prev), out->label = KANN_L_OUT;
	cost = kad_ce2(prev, truth), cost->label = KANN_L_COST;
	a->v = kad_compile(&a->n, 2, out, cost);
	kann_collate_var(a);
	return a;
}

kann_t *kann_rnn_gen_vanilla(int n_in, int n_out, int n_hidden_layers, int n_hidden_neurons)
{
	int i, n_layers, *n_neurons;
	kad_node_t **h_in, **h_out, *t, *truth, *cost = 0;
	kann_t *a;

	a = kann_new();
	n_layers = n_hidden_layers + 2;
	n_neurons = (int*)alloca(n_layers * sizeof(int));
	n_neurons[0] = n_in, n_neurons[n_layers-1] = n_out;
	for (i = 1; i < n_layers - 1; ++i) n_neurons[i] = n_hidden_neurons;
	h_in = (kad_node_t**)alloca(n_layers * sizeof(kad_node_t*));
	h_out = (kad_node_t**)alloca(n_layers * sizeof(kad_node_t*));

	h_out[0] = kad_par(0, 2, 1, n_in), h_out[0]->label = KANN_L_IN;
	for (i = 1; i < n_layers - 1; ++i) {
		h_in[i] = kad_var(0, 0, 2, 1, n_hidden_neurons);
		h_in[i]->x = (float*)calloc(n_hidden_neurons, sizeof(float));
	}
	truth = kad_par(0, 2, 1, n_out), truth->label = KANN_L_TRUTH;
	for (i = 1; i < n_layers; ++i) {
		kad_node_t *w, *u, *b;
		w = kann_new_weight(n_neurons[i], n_neurons[i-1]);
		b = kann_new_bias(n_neurons[i]);
		if (i < n_layers - 1) {
			u = kann_new_weight(n_neurons[i], n_neurons[i]);
			h_out[i] = kad_relu(kad_add(kad_add(kad_cmul(h_out[i-1], w), kad_cmul(h_in[i], u)), b));
			h_out[i]->pre = h_in[i];
		} else {
			t = kad_add(kad_cmul(h_out[i-1], w), b);
			cost = kad_ce2(t, truth), cost->label = KANN_L_COST;
			h_out[i] = kad_sigm(t);
			h_out[i]->label = KANN_L_OUT;
		}
	}
	h_out[n_layers - 2]->label = KANN_L_LAST;
	a->v = kad_compile(&a->n, 2, h_out[n_layers - 1], cost);
	kann_collate_var(a);
	return a;
}

kann_t *kann_rnn_gen_gru(int n_in, int n_out, int n_h_layers, int n_h_neurons)
{
	int i;
	kann_t *a;
	kad_node_t *t, *truth, *cost;
	a = kann_new();
	t = kad_par(0, 2, 1, n_in), t->label = KANN_L_IN;
	for (i = 0; i < n_h_layers; ++i)
		t = kann_layer_gru(t, n_h_neurons);
	t = kann_layer_linear(t, n_out);
	truth = kad_par(0, 2, 1, n_out), truth->label = KANN_L_TRUTH;
	cost = kad_ce2(t, truth), cost->label = KANN_L_COST;
	t = kad_sigm(t), t->label = KANN_L_OUT;
	a->v = kad_compile(&a->n, 2, t, cost);
	kann_collate_var(a);
	return a;
}
