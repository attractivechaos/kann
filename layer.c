#include <assert.h>
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

kad_node_t *kann_layer_input(int n1)
{
	kad_node_t *t;
	t = kad_par(0, 2, 1, n1), t->label = KANN_L_IN;
	return t;
}

kad_node_t *kann_layer_linear(kad_node_t *in, int n1)
{
	int n0;
	kad_node_t *w, *b;
	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	w = kann_new_weight(n1, n0);
	b = kann_new_bias(n1);
	return kad_add(kad_cmul(in, w), b);
}

kad_node_t *kann_layer_dropout(kad_node_t *t, float r)
{
	kad_node_t *s;
	s = kad_par(0, 0), s->label = KANN_H_DROPOUT;
	s->x = (float*)calloc(1, sizeof(float));
	*s->x = r;
	return kad_dropout(t, s);
}

kad_node_t *kann_layer_rnn(kad_node_t *in, int n1, kann_activate_f af)
{
	int n0;
	kad_node_t *h0, *w, *u, *b, *out;
	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	h0 = kad_var(0, 0, 2, 1, n1);
	h0->x = (float*)calloc(n1, sizeof(float));
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	out = af(kad_add(kad_add(kad_cmul(in, w), kad_cmul(h0, u)), b));
	out->pre = h0;
	return out;
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

/**********************
 * Finalize the graph *
 **********************/

kann_t *kann_layer_final(kad_node_t *t, int n_out, int type)
{
	kann_t *a = 0;
	kad_node_t *cost = 0, *truth = 0;
	assert(type == KANN_C_BIN_CE || type == KANN_C_CE);
	truth = kad_par(0, 2, 1, n_out), truth->label = KANN_L_TRUTH;
	t = kann_layer_linear(t, n_out);
	if (type == KANN_C_BIN_CE) {
		cost = kad_ce2(t, truth);
		t = kad_sigm(t);
	} else if (type == KANN_C_CE) {
		kad_node_t *temp;
		temp = kad_par(0, 0), temp->label = KANN_H_TEMP;
		temp->x = (float*)calloc(1, sizeof(float));
		*temp->x = 1.0f;
		cost = kad_cesm(t, truth);
		t = kad_softmax2(t, temp);
	}
	t->label = KANN_L_OUT, cost->label = KANN_L_COST;
	a = (kann_t*)calloc(1, sizeof(kann_t));
	a->v = kad_compile(&a->n, 2, t, cost);
	kann_collate_x(a);
	kad_drand = kann_drand;
	return a;
}
