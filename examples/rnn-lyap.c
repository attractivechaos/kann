#include <unistd.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "kann.h"

float norm_vec(int n, float *t)
{
	int i;
	double s = 0.0;
	for (i = 0; i < n; ++i)
		s += t[i] * t[i];
	s = sqrt(s);
	for (i = 0; i < n; ++i) t[i] /= s;
	return s;
}

int main(int argc, char *argv[])
{
	int i, j, c, l, n = 10000, burn_in = 1000, print_val = 0;
	kad_node_t *p0 = 0;
	float eps = 1e-4, *t, *x, *h0, *h1;
	double lyap = 0.0, lyap_prod = 1.0, lyap_t = 0.0;
	kann_t *ann;
	
	while ((c = getopt(argc, argv, "n:b:e:p")) >= 0) {
		if (c == 'n') n = atoi(optarg);
		else if (c == 'b') burn_in = atoi(optarg);
		else if (c == 'e') eps = atof(optarg);
		else if (c == 'p') print_val = 1;
	}
	if (argc - optind < 1) {
		fprintf(stderr, "Usage: rnn-lyap [options] <in.knm>\n");
		fprintf(stderr, "Options:\n");
		fprintf(stderr, "  -b INT      number of burn-in [%d]\n", burn_in);
		fprintf(stderr, "  -n INT      number of iterations [%d]\n", n);
		fprintf(stderr, "  -e FLOAT    epsilon [%g]\n", eps);
		return 1;
	}

	ann = kann_load(argv[optind]);
	for (i = 0; i < ann->n; ++i) {
		kad_node_t *p = ann->v[i];
		if (p->pre) {
			assert(p0 == 0); // TODO: this means the program would not work with LSTM as it has two recurrent points. It can be relexed with more code
			p0 = p->pre;
		}
	}
	assert(p0 != 0);

	l = kad_len(p0);
	t = (float*)malloc(l * sizeof(float));
	for (i = 0; i < l; ++i)
		t[i] = 2.0 * (kann_drand() - 0.5);
	norm_vec(l, t);

	h0 = (float*)malloc(l * sizeof(float));
	h1 = (float*)malloc(l * sizeof(float));
	memcpy(h0, p0->x, l * sizeof(float));

	x = (float*)calloc(kann_n_in(ann), sizeof(float));
	kann_rnn_start(ann);
	for (j = 0; j < n + burn_in; ++j) {
		float norm;
		for (i = 0; i < l; ++i)
			h1[i] = h0[i] + t[i] * eps;
		// compute the original output
		memcpy(p0->x, h0, l * sizeof(float));
		kann_apply1(ann, x);
		memcpy(h0, p0->x, l * sizeof(float));
		// compute the nearby output
		memcpy(p0->x, h1, l * sizeof(float));
		kann_apply1(ann, x);
		memcpy(h1, p0->x, l * sizeof(float));
		// update the direction vector and the Lyapunov exponent
		for (i = 0; i < l; ++i) t[i] = h1[i] - h0[i];
		norm = norm_vec(l, t) / eps;
		if (j >= burn_in) {
			lyap_prod *= norm, lyap_t += 1.0;
			if (lyap_prod > 1e100 || lyap_prod < 1e-100) {
				lyap += log(lyap_prod);
				lyap_prod = 1.0;
			}
			if (print_val) {
				printf("R\t%d\t%g", j - burn_in, norm);
				for (i = 0; i < l; ++i) printf("\t%g", h0[i]);
				printf("\n");
			}
		}
	}
	lyap = (lyap + log(lyap_prod)) / lyap_t;
	kann_rnn_end(ann);
	free(x);
	printf("L\t%f\n", lyap);

	free(h1); free(h0); free(t);
	kann_delete(ann);
	return 0;
}
