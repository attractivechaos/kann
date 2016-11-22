#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include "kann.h"
#include "kann_rand.h"

static int bit_len = 31, n_train = 10000, n_validate = 1000, n_proc_t, n_proc_v, fixed_train = 0;
static uint64_t *ta, *tb, *tc, *va, *vb, *vc;

static void shuffle_3(size_t n, uint64_t *a, uint64_t *b, uint64_t *c)
{
	int i, j;
	for (i = n; i > 1; --i) {
		uint64_t tmp;
		j = (int)(kann_drand() * i);
		tmp = a[j], a[j] = a[i-1], a[i-1] = tmp;
		tmp = b[j], b[j] = b[i-1], b[i-1] = tmp;
		tmp = c[j], c[j] = c[i-1], c[i-1] = tmp;
	}
}

static inline void gen_num(int bit_len, uint64_t *a, uint64_t *b, uint64_t *c)
{
	uint64_t mask = (1ULL<<bit_len) - 1;
	*a = kann_rand() & mask;
	*b = kann_rand() & mask;
//	*c = (*a + *b) & mask;
	*c = (*a * 53 + *b * 17) & mask;
}

static void num2vec(int bit_len, int k, float **x, float **y, uint64_t a, uint64_t b, uint64_t c)
{
	int j, offx = k * 4, offy = k * 2;
	for (j = 0; j < bit_len; ++j) {
		memset(&x[j][offx], 0, 4 * sizeof(float));
		if (y) memset(&y[j][offy], 0, 2 * sizeof(float));
		x[j][offx + (a>>j&1) + 0] = 1.0f;
		x[j][offx + (b>>j&1) + 2] = 1.0f;
		if (y) y[j][offy + (c>>j&1)] = 1.0f;
	}
}

static void gen_vec(int bit_len, int k, float **x, float **y)
{
	uint64_t a, b, c;
	gen_num(bit_len, &a, &b, &c);
	num2vec(bit_len, k, x, y, a, b, c);
}

static void gen_array(int bit_len)
{
	int i;
	ta = (uint64_t*)realloc(ta, n_train * 8);
	tb = (uint64_t*)realloc(tb, n_train * 8);
	tc = (uint64_t*)realloc(tc, n_train * 8);
	for (i = 0; i < n_train; ++i)
		gen_num(bit_len, &ta[i], &tb[i], &tc[i]);
	va = (uint64_t*)realloc(va, n_validate * 8);
	vb = (uint64_t*)realloc(vb, n_validate * 8);
	vc = (uint64_t*)realloc(vc, n_validate * 8);
	for (i = 0; i < n_validate; ++i)
		gen_num(bit_len, &va[i], &vb[i], &vc[i]);
}

int add_reader(void *data, int action, int *len, int max_bs, float **x, float **y)
{
	if (action == KANN_RA_RESET) {
		n_proc_t = n_proc_v = 0;
		shuffle_3(n_train, ta, tb, tc);
	} else if (action == KANN_RA_READ_TRAIN || action == KANN_RA_READ_VALIDATE) {
		int k;
		if (action == KANN_RA_READ_TRAIN    && n_proc_t + max_bs > n_train)    return 0;
		if (action == KANN_RA_READ_VALIDATE && n_proc_v + max_bs > n_validate) return 0;
		if (fixed_train) {
			if (action == KANN_RA_READ_TRAIN) {
				int b = n_proc_t + max_bs < n_train? max_bs : n_train - n_proc_t;
				for (k = 0; k < b; ++k) num2vec(bit_len, k, x, y, ta[n_proc_t+k], tb[n_proc_t+k], tc[n_proc_t+k]);
				n_proc_t += b;
				return b;
			} else {
				int b = n_proc_v + max_bs < n_validate? max_bs : n_validate - n_proc_v;
				for (k = 0; k < b; ++k) num2vec(bit_len, k, x, y, va[n_proc_v+k], vb[n_proc_v+k], vc[n_proc_v+k]);
				n_proc_v += b;
				return b;
			}
		} else {
			for (k = 0; k < max_bs; ++k) gen_vec(bit_len, k, x, y);
			if (action == KANN_RA_READ_TRAIN) n_proc_t += max_bs;
			else n_proc_v += max_bs;
			return max_bs;
		}
	}
	return 0;
}

int main(int argc, char *argv[])
{
	kann_t *ann;
	kann_mopt_t mo;
	float **x, *y;
	int i, k, t, use_gru = 1, n_h_layers = 1, n_h_neurons = 50, n_print = 10;
	uint64_t seed = 11;
	char *fn_out = 0, *fn_in = 0;

	kann_mopt_init(&mo);
	mo.max_epoch = 50;
	while ((t = getopt(argc, argv, "m:b:l:r:hn:vo:i:s:t:f")) >= 0) {
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
		else if (t == 'f') fixed_train = 1;
		else if (t == 'h') {
			FILE *fp = stdout;
			fprintf(fp, "Usage: rnn-add [options]\nOptions:\n");
			fprintf(fp, "  -i FILE    load a trained model []\n");
			fprintf(fp, "  -o FILE    save the trained model []\n");
			fprintf(fp, "  -s INT     random seed [%ld]\n", (long)seed);
			fprintf(fp, "  -l INT     number of hidden layers [%d]\n", n_h_layers);
			fprintf(fp, "  -n INT     number of neurons at each hidden layer [%d]\n", n_h_neurons);
			fprintf(fp, "  -m INT     max epoch [%d]\n", mo.max_epoch);
			fprintf(fp, "  -r FLOAT   learning rate [%g]\n", mo.lr);
			fprintf(fp, "  -b INT     number of bits [%d]\n", bit_len);
			fprintf(fp, "  -f         use a fixed training/validation set\n");
			fprintf(fp, "  -v         use vanilla RNN (GRU by default)\n");
			return 1;
		}
	}

	kann_srand(seed);
	gen_array(bit_len);
	mo.max_rnn_len = bit_len;

	if (fn_in) ann = kann_read(fn_in);
	else ann = use_gru? kann_rnn_gen_gru(4, 2, n_h_layers, n_h_neurons) : kann_rnn_gen_vanilla(4, 2, n_h_layers, n_h_neurons);
	kann_train(&mo, ann, add_reader, 0);
	if (fn_out) kann_write(fn_out, ann);

	x = (float**)calloc(bit_len, sizeof(float*));
	for (i = 0; i < bit_len; ++i)
		x[i] = (float*)calloc(4, sizeof(float));
	for (k = 0; k < n_print; ++k) {
		uint64_t a, b, c, d;
		gen_num(bit_len, &a, &b, &c);
		num2vec(bit_len, 0, x, 0, a, b, c);
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
	}

	kann_delete(ann);
	return 0;
}
