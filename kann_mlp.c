#include <stdlib.h>
#include <string.h>
#include "kann_rand.h"
#include "kann_ann.h"
#include "kann_mlp.h"
/*
kann_mlp_t *kann_mlp_init(int n_in, int n_out, int n_hidden_layers, int n_hidden_neurons)
{
	int i, j, n_par = 0;
	kad_node_t *in, *rt, *rp, *out, *prev;
	kann_mlp_t *m;

	m = (kann_mlp_t*)calloc(1, sizeof(kann_mlp_t));
	m->n_layers = n_hidden_layers + 2;
	m->n_neurons = (int*)malloc(m->n_layers * sizeof(int));
	m->n_neurons[0] = n_in, m->n_neurons[m->n_layers-1] = n_out;
	for (i = 1; i < m->n_layers - 1; ++i) m->n_neurons[i] = n_hidden_neurons;
	for (i = 1; i < m->n_layers; ++i)
		n_par += m->n_neurons[i] * (m->n_neurons[i-1] + 1);
	m->t = (float*)calloc(n_par, sizeof(float));
	m->g = (float*)calloc(n_par, sizeof(float));
	m->kr = kann_srand_r(11);

	prev = in = kad_par(1, n_in, 0);
	in->label = KAD_LABEL_INPUT;
	out = kad_par(1, n_out, 0);
	out->label = KAD_LABEL_OUTPUT;
	for (i = 1, j = 0; i < m->n_layers; ++i) {
		kad_node_t *w, *b;
		if (i > 1) prev = kad_relu(prev);
		w = kad_var(m->n_neurons[i], m->n_neurons[i-1], &m->t[j], &m->g[j]);
		kann_rand_weight(m->kr, m->n_neurons[i], m->n_neurons[i-1], &m->t[j]);
		j += m->n_neurons[i] * m->n_neurons[i-1];
		b = kad_var(1, m->n_neurons[i], &m->t[j], &m->g[j]);
		j += m->n_neurons[i];
		prev = kad_add(kad_mtmul(prev, w), b);
	}
	rt = kad_ce2(prev, out);
	rp = kad_sigm(prev);
	m->mt = kad_compile(rt, &m->n_mt);
	m->mp = kad_compile(rp, &m->n_mp);
	return m;
}

void kann_mlp_destroy(kann_mlp_t *m)
{
}

void kann_mlp_gradient(int n, const float *x, float *g, void *data)
{
	kann_mlp_t *m = (kann_mlp_t*)data;
	memcpy(m->t, x, n * sizeof(float));
	kad_eval(m->n_mt, m->mt, 1);
	memcpy(g, m->g, n * sizeof(float));
}
*/
