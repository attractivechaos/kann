#include "kann_rand.h"
#include "kann.h"

static kann_t *kann_gen_common(int n_in, int n_out, int n_h_layers, int n_h_neurons, kann_layer_f layer)
{
	int i;
	kann_t *a;
	kad_node_t *t, *truth, *cost;
	a = kann_new();
	t = kad_par(0, 2, 1, n_in), t->label = KANN_L_IN;
	for (i = 0; i < n_h_layers; ++i)
		t = layer(t, n_h_neurons);
	t = kann_layer_linear(t, n_out);
	truth = kad_par(0, 2, 1, n_out), truth->label = KANN_L_TRUTH;
	cost = kad_ce2(t, truth), cost->label = KANN_L_COST;
	t = kad_sigm(t), t->label = KANN_L_OUT;
	a->v = kad_compile(&a->n, 2, t, cost);
	kann_collate_var(a);
	return a;
}

kann_t *kann_fnn_gen_mlp(int n_in, int n_out, int n_h_layers, int n_h_neurons)
{
	return kann_gen_common(n_in, n_out, n_h_layers, n_h_neurons, kann_layer_linear_relu);
}

kann_t *kann_rnn_gen_vanilla(int n_in, int n_out, int n_h_layers, int n_h_neurons)
{
	return kann_gen_common(n_in, n_out, n_h_layers, n_h_neurons, kann_layer_rnn);
}

kann_t *kann_rnn_gen_gru(int n_in, int n_out, int n_h_layers, int n_h_neurons)
{
	return kann_gen_common(n_in, n_out, n_h_layers, n_h_neurons, kann_layer_gru);
}
