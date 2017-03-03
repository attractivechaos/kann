#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
#include "kann.h"

typedef struct {
	int n_in, ulen;
	int n, m;
	uint64_t *x, *y;
} bit_data_t;

#define MAX_FIELDS 64

static int read_int(FILE *fp, uint64_t x[MAX_FIELDS])
{
	char *p, *q, line[1024];
	int i;
	if (feof(fp) || fgets(line, 1024, fp) == 0) return 0;
	for (q = p = line, i = 0; *p; ++p) {
		if (isspace(*p)) {
			long t;
			t = strtol(q, &q, 10);
			assert(t >= 0);
			x[i++] = t;
			if (i == MAX_FIELDS) break;
			q = p + 1;
		}
	}
	return i;
}

static bit_data_t *read_data(const char *fn)
{
	bit_data_t *d;
	FILE *fp;
	int i, j;
	uint64_t max, x[MAX_FIELDS];

	fp = fn && strcmp(fn, "-")? fopen(fn, "r") : stdin;
	if (fp == 0) return 0;
	d = (bit_data_t*)calloc(1, sizeof(bit_data_t));
	while ((i = read_int(fp, x)) > 0) {
		assert(d->n == 0 || d->n_in == i - 1);
		d->n_in = i - 1;
		if (d->n == d->m) {
			d->m = d->m? d->m<<1 : 256;
			d->x = (uint64_t*)realloc(d->x, d->m * d->n_in * 8);
			d->y = (uint64_t*)realloc(d->y, d->m * 8);
		}
		memcpy(&d->x[d->n * d->n_in], x, d->n_in * 8);
		d->y[d->n++] = x[d->n_in];
	}
	fclose(fp);
	for (i = 0, max = 0; i < d->n; ++i) {
		int t = i * d->n_in;
		for (j = 0; j < d->n_in; ++j)
			max = max > d->x[t + j]? max : d->x[t + j];
		max = max > d->y[i]? max : d->y[i];
	}
	for (i = 0; max; max >>= 1, ++i);
	d->ulen = i;
	return d;
}

static void train(kann_t *ann, bit_data_t *d, float lr, int mini_size, int max_epoch, const char *fn, int n_threads)
{
	float **x, **y, *r, best_cost = 1e30f;
	int epoch, j, n_var, *shuf;
	kann_t *ua;

	n_var = kann_size_var(ann);
	r = (float*)calloc(n_var, sizeof(float));
	x = (float**)malloc(d->ulen * sizeof(float*));
	y = (float**)malloc(d->ulen * sizeof(float*));
	for (j = 0; j < d->ulen; ++j) {
		x[j] = (float*)calloc(mini_size * d->n_in, sizeof(float));
		y[j] = (float*)calloc(mini_size * 2, sizeof(float));
	}
	shuf = (int*)calloc(d->n, sizeof(int));
	kann_shuffle(d->n, shuf);

	ua = kann_unroll(ann, d->ulen);
	kann_set_batch_size(ua, mini_size);
	kann_mt(ua, n_threads, mini_size);
	kann_feed_bind(ua, KANN_F_IN,    0, x);
	kann_feed_bind(ua, KANN_F_TRUTH, 0, y);
	kann_switch(ua, 1);
	for (epoch = 0; epoch < max_epoch; ++epoch) {
		double cost = 0.0;
		int tot = 0, tot_base = 0, n_cerr = 0;
		for (j = 0; j < d->n - mini_size; j += mini_size) {
			int i, b, k;
			for (k = 0; k < d->ulen; ++k) {
				for (b = 0; b < mini_size; ++b) {
					int s = shuf[j + b];
					for (i = 0; i < d->n_in; ++i)
						x[k][b * d->n_in + i] = (float)(d->x[s * d->n_in + i] >> k & 1);
					y[k][b * 2] = y[k][b * 2 + 1] = 0.0f;
					y[k][b * 2 + (d->y[s] >> k & 1)] = 1.0f;
				}
			}
			cost += kann_cost(ua, 0, 1) * d->ulen * mini_size;
			n_cerr += kann_class_error(ua, &k);
			tot_base += k;
			//kad_check_grad(ua->n, ua->v, ua->n-1);
			kann_RMSprop(n_var, lr, 0, 0.9f, ua->g, ua->x, r);
			tot += d->ulen * mini_size;
		}
		if (cost < best_cost) {
			best_cost = cost;
			if (fn) kann_save(fn, ann);
		}
		fprintf(stderr, "epoch: %d; cost: %g (class error: %.2f%%)\n", epoch+1, cost / tot, 100.0f * n_cerr / tot_base);
	}

	for (j = 0; j < d->ulen; ++j) {
		free(y[j]); free(x[j]);
	}
	free(y); free(x); free(r); free(shuf);
}

int main(int argc, char *argv[])
{
	int i, c, seed = 11, n_h_layers = 1, n_h_neurons = 64, mini_size = 64, max_epoch = 50, to_apply = 0, norm = 1, n_threads = 1;
	float lr = 0.01f, dropout = 0.2f;
	kann_t *ann = 0;
	char *fn_in = 0, *fn_out = 0;

	while ((c = getopt(argc, argv, "i:o:l:n:m:r:s:Ad:Nt:")) >= 0) {
		if (c == 'i') fn_in = optarg;
		else if (c == 'o') fn_out = optarg;
		else if (c == 'l') n_h_layers = atoi(optarg);
		else if (c == 'n') n_h_neurons = atoi(optarg);
		else if (c == 'm') max_epoch = atoi(optarg);
		else if (c == 'r') lr = atof(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 'A') to_apply = 1;
		else if (c == 'N') norm = 0;
		else if (c == 'd') dropout = atof(optarg);
		else if (c == 't') n_threads = atoi(optarg);
	}
	if (optind == argc) {
		fprintf(stderr, "Usage: rnn-bit [options] <in.txt>\n");
		return 1;
	}
	kad_trap_fe();
	kann_srand(seed);
	if (fn_in) ann = kann_load(fn_in);

	if (!to_apply) {
		bit_data_t *d;
		d = read_data(argv[optind]);
		if (ann == 0) { // model generation
			kad_node_t *t;
			int rnn_flag = KANN_RNN_VAR_H0;
			if (norm) rnn_flag |= KANN_RNN_NORM;
			t = kann_layer_input(d->n_in);
			for (i = 0; i < n_h_layers; ++i) {
				t = kann_layer_gru(t, n_h_neurons, rnn_flag);
				t = kann_layer_dropout(t, dropout);
			}
			ann = kann_new(kann_layer_cost(t, 2, KANN_C_CEM), 0);
		}
		train(ann, d, lr, mini_size, max_epoch, fn_out, n_threads);
		free(d->x); free(d->y); free(d);
	} else {
		FILE *fp;
		uint64_t x[MAX_FIELDS], y;
		int n, i, k, n_in;
		n_in = kann_dim_in(ann);
		fp = strcmp(argv[optind], "-")? fopen(argv[optind], "r") : stdin;
		while ((n = read_int(fp, x)) > 0) {
			float x1[MAX_FIELDS];
			assert(n >= n_in);
			kann_rnn_start(ann);
			for (k = 0, y = 0; k < 64; ++k) {
				const float *y1;
				for (i = 0; i < n_in; ++i)
					x1[i] = (float)(x[i] >> k & 1);
				y1 = kann_apply1(ann, x1);
				if (y1[1] > y1[0]) y |= 1ULL << k;
			}
			kann_rnn_end(ann);
			printf("%llu\n", (unsigned long long)y);
		}
		fclose(fp);
	}

	kann_delete(ann);
	return 0;
}
