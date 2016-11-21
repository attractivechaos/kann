#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include "kann.h"
#include "kann_rand.h"

static int bit_len = 31, n_train = 10000, n_validate = 1000, n_proc_t, n_proc_v;

static void gen_addition(int k, float **x, float **y, uint64_t *_a, uint64_t *_b, uint64_t *_c)
{
	uint64_t a, b, c, mask = (1ULL<<bit_len) - 1;
	int j, offx = k * 4, offy = k * 2;
	a = kann_rand() & mask;
	b = kann_rand() & mask;
//	c = ((a ^ b<<5) - 50) & mask;
	c = (a + b) & mask;
//	c = (a * 13612 + b) & mask;
//	c = (a * 50 + b) & mask;
	for (j = 0; j < bit_len; ++j) {
		memset(&x[j][offx], 0, 4 * sizeof(float));
		if (y) memset(&y[j][offy], 0, 2 * sizeof(float));
		x[j][offx + (a>>j&1) + 0] = 1.0f;
		x[j][offx + (b>>j&1) + 2] = 1.0f;
		if (y) y[j][offy + (c>>j&1)] = 1.0f;
	}
	if (_a) *_a = a;
	if (_b) *_b = b;
	if (_c) *_c = c;
}

int add_reader(void *data, int action, int *len, int max_bs, float **x, float **y)
{
	if (action == KANN_RA_RESET) {
		n_proc_t = n_proc_v = 0;
	} else if (action == KANN_RA_READ_TRAIN || action == KANN_RA_READ_VALIDATE) {
		int k;
		if (action == KANN_RA_READ_TRAIN    && n_proc_t + max_bs > n_train)    return 0;
		if (action == KANN_RA_READ_VALIDATE && n_proc_v + max_bs > n_validate) return 0;
		for (k = 0; k < max_bs; ++k)
			gen_addition(k, x, y, 0, 0, 0);
		if (action == KANN_RA_READ_TRAIN) n_proc_t += max_bs;
		else n_proc_v += max_bs;
		return max_bs;
	}
	return 0;
}

int main(int argc, char *argv[])
{
	kann_t *ann;
	kann_mopt_t mo;
	float **x, *y;
	int i, t, use_gru = 0, n_h_layers = 1, n_h_neurons = 20;
	uint64_t a, b, c, d;

	kann_mopt_init(&mo);
	mo.max_epoch = 20;
	while ((t = getopt(argc, argv, "m:b:gl:r:hn:")) >= 0) {
		if (t == 'm') mo.max_epoch = atoi(optarg);
		else if (t == 'b') bit_len = atoi(optarg);
		else if (t == 'r') mo.lr = atof(optarg);
		else if (t == 'g') use_gru = 1;
		else if (t == 'l') n_h_layers = atoi(optarg);
		else if (t == 'n') n_h_neurons = atoi(optarg);
		else if (t == 'h') {
			FILE *fp = stdout;
			fprintf(fp, "Usage: rnn-add [options]\nOptions:\n");
			fprintf(fp, "  -l INT     number of hidden layers [%d]\n", n_h_layers);
			fprintf(fp, "  -n INT     number of neurons at each hidden layer [%d]\n", n_h_neurons);
			fprintf(fp, "  -m INT     max epoch [%d]\n", mo.max_epoch);
			fprintf(fp, "  -r FLOAT   learning rate [%g]\n", mo.lr);
			fprintf(fp, "  -b INT     number of bits [%d]\n", bit_len);
			fprintf(fp, "  -g         use GRU (vanilla RNN by default)\n");
			return 1;
		}
	}

	mo.max_rnn_len = bit_len;
	ann = use_gru? kann_rnn_gen_gru(4, 2, n_h_layers, n_h_neurons) : kann_rnn_gen_vanilla(4, 2, n_h_layers, n_h_neurons);
	kann_train(&mo, ann, add_reader, 0);

	x = (float**)calloc(bit_len, sizeof(float*));
	for (i = 0; i < bit_len; ++i)
		x[i] = (float*)calloc(4, sizeof(float));
	gen_addition(0, x, 0, &a, &b, &c);
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

	kann_delete(ann);
	return 0;
}
