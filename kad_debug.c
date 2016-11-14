#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "kautodiff.h"

void kad_debug(FILE *fp, int n, kad_node_t **v)
{
	static const char *op[] = { "", "add", "mul", "cmul", "ce2", "norm2", "sigm", "tanh", "relu" };
	int i, j;
	for (i = 0; i < n; ++i) v[i]->tmp = i;
	for (i = 0; i < n; ++i) {
		kad_node_t *p = v[i];
		fprintf(stderr, "%d\t", i);
		if (p->n_child) {
			fprintf(fp, "%s(", op[p->op]);
			for (j = 0; j < p->n_child; ++j) {
				if (j) fputc(',', fp);
				fprintf(fp, "$%d", p->child[j].p->tmp);
			}
			fputc(')', fp);
		} else fprintf(fp, "%c|%d", p->to_back? 'v' : 'p', p->label);
		fputs("\t[", fp);
		for (j = 0; j < p->n_d; ++j) {
			if (j) fputc(',', fp);
			fprintf(fp, "%d", p->d[j]);
		}
		fputc(']', fp);
		fputc('\n', fp);
	}
	for (i = 0; i < n; ++i) v[i]->tmp = 0;
}

static void kad_add_delta(int n, kad_node_t **a, float c, float *delta)
{
	int i, k;
	for (i = k = 0; i < n; ++i)
		if (kad_is_var(a[i])) {
			kad_saxpy(kad_len(a[i]), c, &delta[k], a[i]->x);
			k += kad_len(a[i]);
		}
}

void kad_check_grad(int n, kad_node_t **a, int from)
{
	const float eps = 1e-5, rel = .01f;
	int i, k, n_var;
	float *g0, *delta, f0, f_minus, f_plus, s0, s1, rel_err, p_m_err;
	n_var = kad_n_var(n, a);
	g0 = (float*)calloc(n_var, sizeof(float));
	f0 = *kad_eval(n, a, from);
	kad_grad(n, a, from);
	for (i = k = 0; i < n; ++i)
		if (kad_is_var(a[i])) {
			memcpy(&g0[k], a[i]->g, kad_len(a[i]) * sizeof(float));
			k += kad_len(a[i]);
		}
	delta = (float*)calloc(n_var, sizeof(float));
	for (k = 0; k < n_var; ++k) delta[k] = drand48() * eps;
	kad_add_delta(n, a, 1.0f, delta);
	f_plus = *kad_eval(n, a, from);
	kad_add_delta(n, a, -2.0f, delta);
	f_minus = *kad_eval(n, a, from);
	kad_add_delta(n, a, 1.0f, delta);
	s0 = kad_sdot(n_var, g0, delta);
	s1 = .5 * (f_plus - f_minus);
	if (fabs(s0) >= rel * eps && fabs(s1) >= rel * eps) {
		rel_err = fabs(fabs(s0) - fabs(s1)) / (fabs(s0) + fabs(s1));
		p_m_err = fabs(f_plus + f_minus - 2.0f * f0) / fabs(f_plus - f_minus);
		if (rel_err >= rel && rel_err > p_m_err)
			fprintf(stderr, "%g,%g,%g\n", s0/eps, rel_err, p_m_err);
	}
	free(delta); free(g0);
}
