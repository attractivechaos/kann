#include <unistd.h>
#include <stdlib.h>
#include "kann.h"

#define RN_N_IN  2
#define RN_N_OUT 1

#define rn_cal(a, b) ((a) * 53 + (b) * 17)

typedef struct {
	int tot[2], proc[2];
} reader_conf_t;

static inline void num2vec(int len, int step, uint64_t a, float *x) // convert a number to a bit vector
{
	int i, j;
	for (i = j = 0; i < len; ++i, j += step*2)
		x[j] = x[j+1] = 0.0f, x[j + (a>>i&1)] = 1.0f;
}

static int data_reader(void *data, int action, int len, float *x, float *y) // callback to feed data to RNN
{
	reader_conf_t *g = (reader_conf_t*)data;
	if (action == KANN_RDR_BATCH_RESET) {
		g->proc[0] = g->proc[1] = 0;
	} else if (action == KANN_RDR_READ_TRAIN || action == KANN_RDR_READ_VALIDATE) {
		int k = action == KANN_RDR_READ_TRAIN? 0 : 1;
		if (g->proc[k] < g->tot[k]) {
			uint64_t a, b;
			a = kann_rand(), b = kann_rand();
			num2vec(len, RN_N_IN,  a, x);
			num2vec(len, RN_N_IN,  b, x+2);
			num2vec(len, RN_N_OUT, rn_cal(a,b), y);
			++g->proc[k];
		} else return 0;
	}
	return len;
}

int main(int argc, char *argv[])
{
	reader_conf_t conf = { {10000,1000}, {0,0} };
	int i, c, n_err, seed = 11, n_h_layers = 1, n_h_neurons = 64, len = 30;
	kann_t *ann;
	kann_mopt_t mo;
	char *fn_in = 0, *fn_out = 0;

	kann_mopt_init(&mo);
	mo.lr = 0.01f;
	mo.max_epoch = 100;
	while ((c = getopt(argc, argv, "i:o:l:h:m:r:")) >= 0) {
		if (c == 'i') fn_in = optarg;
		else if (c == 'o') fn_out = optarg;
		else if (c == 'l') n_h_layers = atoi(optarg);
		else if (c == 'h') n_h_neurons = atoi(optarg);
		else if (c == 'm') mo.max_epoch = atoi(optarg);
		else if (c == 'r') mo.lr = atof(optarg);
	}

	kann_srand(seed);
	mo.max_rnn_len = len;
	if (fn_in) { // then read the network from file
		ann = kann_read(fn_in);
	} else { // model generation
		kad_node_t *t;
		t = kann_layer_input(RN_N_IN*2);
		for (i = 0; i < n_h_layers; ++i)
			t = kann_layer_gru(t, n_h_neurons);
		ann = kann_layer_final(t, RN_N_OUT*2, KANN_C_CEB);
	}
	kann_train(&mo, ann, data_reader, &conf);
	if (fn_out) kann_write(fn_out, ann);

	for (i = n_err = 0; i < conf.tot[1]; ++i) { // apply to 64-bit integers for testing
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
	fprintf(stderr, "Error on 64-bit integers: %.2f%%\n", 100.0f * n_err / conf.tot[1] / 64);

	kann_delete(ann);
	return 0;
}
