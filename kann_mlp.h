#ifndef KANN_MLP_H
#define KANN_MLP_H

#include "kautodiff.h"

typedef struct {
	int n_mt, n_mp, n_layers, *n_neurons;
	kad_node_t **mt, **mp;
	float *t, *g;
	void *kr;
} kann_mlp_t;

#endif
