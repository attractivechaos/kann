#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include "kann.h"
#include "kann_rand.h"

static int bit_len = 31, n_tot[2] = {10000,1000}, n_proc[2];
static uint64_t *ta, *tb, *tc;

static inline void gen_num(int bit_len, uint64_t *a, uint64_t *b, uint64_t *c)
{
	uint64_t mask = (1ULL<<bit_len) - 1;
	*a = kann_rand() & mask;
	*b = kann_rand() & mask;
//	*c = (*a + *b) & mask;
	*c = (*a * 53 + *b * 17) & mask;
}

static void num2vec(int bit_len, float *x, float *y, uint64_t a, uint64_t b, uint64_t c)
{
	int j;
	float *x1, *y1;
	for (j = 0; j < bit_len; ++j) {
		x1 = x + j * 4, y1 = y + j * 2;
		memset(x1, 0, 4 * sizeof(float));
		if (y) memset(y1, 0, 2 * sizeof(float));
		x1[(a>>j&1) + 0] = 1.0f;
		x1[(b>>j&1) + 2] = 1.0f;
		if (y) y1[c>>j&1]= 1.0f;
	}
}

int bit_reader(void *data, int action, int len, float *x, float *y)
{
	int i, j;
	if (action == KANN_RDR_BATCH_RESET) {
		n_proc[0] = n_proc[1] = 0;
		for (i = n_tot[0]; i > 1; --i) {
			uint64_t tmp;
			j = (int)(kann_drand() * i);
			tmp = ta[j], ta[j] = ta[i-1], ta[i-1] = tmp;
			tmp = tb[j], tb[j] = tb[i-1], tb[i-1] = tmp;
			tmp = tc[j], tc[j] = tc[i-1], tc[i-1] = tmp;
		}
	} else if (action == KANN_RDR_READ_TRAIN || action == KANN_RDR_READ_VALIDATE) {
		i = action == KANN_RDR_READ_TRAIN? 0 : 1;
		j = i? n_tot[0] : 0;
		if (n_proc[i] < n_tot[i]) {
			num2vec(bit_len, x, y, ta[n_proc[i]+j], tb[n_proc[i]+j], tc[n_proc[i]+j]);
			++n_proc[i];
		} else return 0;
	}
	return bit_len;
}

int main(int argc, char *argv[])
{
	kann_t *ann;
	kann_mopt_t mo;
	float *x, *y;
	int i, k, t, use_gru = 1, n_h_layers = 1, n_h_neurons = 50, n_print = 10;
	uint64_t seed = 131;
	char *fn_out = 0, *fn_in = 0;

	kann_mopt_init(&mo);
	mo.max_epoch = 50;
	while ((t = getopt(argc, argv, "m:b:l:r:hn:vo:i:s:t:")) >= 0) {
		if (t == 'm') mo.max_epoch = atoi(optarg);
		else if (t == 'b') bit_len = atoi(optarg);
		else if (t == 'r') mo.lr = atof(optarg);
		else if (t == 'v') use_gru = 0;
		else if (t == 'l') n_h_layers = atoi(optarg);
		else if (t == 'n') n_h_neurons = atoi(optarg);
		else if (t == 'o') fn_out = optarg;
		else if (t == 'i') fn_in = optarg;
		else if (t == 't') n_print = atoi(optarg);
		else if (t == 's') seed = atol(optarg);
		else if (t == 'h') {
			FILE *fp = stdout;
			fprintf(fp, "Usage: rnn-bit [options]\nOptions:\n");
			fprintf(fp, "  -i FILE    load a trained model []\n");
			fprintf(fp, "  -o FILE    save the trained model []\n");
			fprintf(fp, "  -s INT     random seed [%ld]\n", (long)seed);
			fprintf(fp, "  -l INT     number of hidden layers [%d]\n", n_h_layers);
			fprintf(fp, "  -n INT     number of neurons at each hidden layer [%d]\n", n_h_neurons);
			fprintf(fp, "  -m INT     max epoch [%d]\n", mo.max_epoch);
			fprintf(fp, "  -r FLOAT   learning rate [%g]\n", mo.lr);
			fprintf(fp, "  -b INT     number of bits [%d]\n", bit_len);
			fprintf(fp, "  -v         use vanilla RNN (GRU by default)\n");
			return 1;
		}
	}

	kann_srand(seed);
	ta = (uint64_t*)malloc((n_tot[0] + n_tot[1]) * 8);
	tb = (uint64_t*)malloc((n_tot[0] + n_tot[1]) * 8);
	tc = (uint64_t*)malloc((n_tot[0] + n_tot[1]) * 8);
	for (i = 0; i < n_tot[0] + n_tot[1]; ++i)
		gen_num(bit_len, &ta[i], &tb[i], &tc[i]);

	if (fn_in) {
		ann = kann_read(fn_in);
	} else {
		kad_node_t *t;
		t = kann_layer_input(bit_len * 4);
		for (i = 0; i < n_h_layers; ++i)
			t = use_gru? kann_layer_gru(t, n_h_neurons) : kann_layer_rnn(t, n_h_neurons);
		ann = kann_layer_final(t, bit_len * 2, KANN_C_BIN_CE);
	}
	mo.max_rnn_len = bit_len;
	kann_train(&mo, ann, bit_reader, 0);
	if (fn_out) kann_write(fn_out, ann);

	x = (float*)calloc(bit_len * 4, sizeof(float));
	for (k = 0; k < n_print; ++k) {
		uint64_t a, b, c, d;
		gen_num(bit_len, &a, &b, &c);
		num2vec(bit_len, x, 0, a, b, c);
		y = kann_rnn_apply_seq1(ann, bit_len, x);
		for (i = 0, d = 0; i < bit_len; ++i) {
			int k = y[i*2] > y[i*2+1]? 0 : 1;
			d |= (uint64_t)k << i;
			putchar('0' + k);
		}
		putchar('\n');
		for (i = 0; i < bit_len; ++i) {
			int k = c>>i&1;
			putchar('0' + k);
		}
		putchar('\n');
		printf("f(%ld,%ld) = %ld ?=? %ld\n", (long)a, (long)b, (long)c, (long)d);
		free(y);
	}

	free(x);
	kann_delete(ann);
	return 0;
}
