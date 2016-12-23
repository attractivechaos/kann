#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "kann.h"

#define RN_N_IN  2
#define RN_N_OUT 1

#define rn_cal(a, b) ((a) * 53 + (b) * 17)

static void train(kann_t *ann, int ulen, float lr, int mini_size, int max_epoch, int n)
{
	float **x, **y, *r;
	int i, j, n_var;
	kann_t *ua;

	n_var = kad_n_var(ann->n, ann->v);
	r = (float*)calloc(n_var, sizeof(float));
	x = (float**)malloc(ulen * sizeof(float*));
	y = (float**)malloc(ulen * sizeof(float*));
	for (j = 0; j < ulen; ++j) {
		x[j] = (float*)calloc(mini_size * RN_N_IN  * 2, sizeof(float));
		y[j] = (float*)calloc(mini_size * RN_N_OUT * 2, sizeof(float));
	}

	ua = kann_unroll(ann, ulen);
	kann_set_batch_size(ua, mini_size);
	kann_bind_feed(ua, KANN_F_IN,    0, x);
	kann_bind_feed(ua, KANN_F_TRUTH, 0, y);
	for (i = 0; i < max_epoch; ++i) {
		double cost = 0.0;
		int tot = 0, n_cerr = 0;
		for (j = 0; j < n; j += mini_size) {
			int m, k;
			for (k = 0; k < ulen; ++k) {
				memset(x[k], 0, mini_size * RN_N_IN  * 2 * sizeof(float));
				memset(y[k], 0, mini_size * RN_N_OUT * 2 * sizeof(float));
			}
			for (m = 0; m < mini_size; ++m) {
				uint64_t a, b, c;
				a = kann_rand(), b = kann_rand();
				c = rn_cal(a, b);
				for (k = 0; k < ulen; ++k) {
					x[k][m * RN_N_IN  * 2 + (a>>k&1)] = 1.0f;
					x[k][m * RN_N_IN  * 2 + 2 + (b>>k&1)] = 1.0f;
					y[k][m * RN_N_OUT * 2 + (c>>k&1)] = 1.0f;
				}
			}
			cost += kann_cost(ua, 1) * ulen * mini_size;
			n_cerr += kann_class_error(ua);
			kann_RMSprop(n_var, lr, 0, 0.9f, ua->g, ua->x, r);
			tot += ulen * mini_size;
		}
		fprintf(stderr, "epoch: %d; cost: %g (class error: %.2f%%)\n", i+1, cost / tot, 100.0f * n_cerr / tot);
	}

	for (j = 0; j < ulen; ++j) {
		free(y[j]); free(x[j]);
	}
	free(y); free(x); free(r);
}

int main(int argc, char *argv[])
{
	int i, c, n_err, seed = 11, n_h_layers = 1, n_h_neurons = 64, mini_size = 64, ulen = 30, max_epoch = 50, N = 10000;
	float lr = 0.01f;
	kann_t *ann;
	char *fn_in = 0, *fn_out = 0;

	while ((c = getopt(argc, argv, "i:o:l:n:m:r:s:u:N:")) >= 0) {
		if (c == 'i') fn_in = optarg;
		else if (c == 'o') fn_out = optarg;
		else if (c == 'l') n_h_layers = atoi(optarg);
		else if (c == 'n') n_h_neurons = atoi(optarg);
		else if (c == 'm') max_epoch = atoi(optarg);
		else if (c == 'r') lr = atof(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 'u') ulen = atoi(optarg);
		else if (c == 'N') N = atoi(optarg);
	}

	kann_srand(seed);
	if (fn_in) { // then read the network from file
		ann = kann_load(fn_in);
	} else { // model generation
		kad_node_t *t;
		t = kann_layer_input(RN_N_IN*2);
		for (i = 0; i < n_h_layers; ++i)
			t = kann_layer_gru(t, n_h_neurons);
		ann = kann_layer_final(t, RN_N_OUT*2, KANN_C_CEB);
	}
	train(ann, ulen, lr, mini_size, max_epoch, N);
	if (fn_out) kann_save(fn_out, ann);

	for (i = n_err = 0; i < N/10; ++i) { // apply to 64-bit integers for testing
		uint64_t a, b, c;
		int j, k;
		a = kann_rand(), b = kann_rand(), c = rn_cal(a, b);
		kann_rnn_start(ann);
		for (j = 0; j < 64; ++j) { // run prediction bit by bit
			float x[RN_N_IN*2];
			const float *y;
			x[0] = x[1] = x[2] = x[3] = 0.0f;
			x[a>>j&1] = 1.0f, x[(b>>j&1)+2] = 1.0f;
			y = kann_apply1(ann, x);
			k = y[0] > y[1]? 0 : 1;
			if (k != (c>>j&1)) ++n_err;
		}
		kann_rnn_end(ann);
	}
	fprintf(stderr, "Error on 64-bit integers: %.2f%%\n", 100.0f * n_err / i / 64);

	kann_delete(ann);
	return 0;
}
