#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "kann.h"
#include "kann_rand.h"

static int bit_len = 31, n_train = 10000, n_validate = 1000, n_proc_t, n_proc_v;

static void gen_addition(int k, float **x, float **y, uint64_t *_a, uint64_t *_b)
{
	uint64_t a, b, c, mask = (1ULL<<bit_len) - 1;
	int j, offx = k * 4, offy = k * 2;
	a = kann_rand() & mask;
	b = kann_rand() & mask;
	c = (a + b) & mask;
	for (j = 0; j < bit_len; ++j) {
		memset(&x[j][offx], 0, 4 * sizeof(float));
		if (y) memset(&y[j][offy], 0, 2 * sizeof(float));
		x[j][offx + (a>>j&1) + 0] = 1.0f;
		x[j][offx + (b>>j&1) + 2] = 1.0f;
		if (y) y[j][offy + (c>>j&1)] = 1.0f;
	}
	if (_a) *_a = a;
	if (_b) *_b = b;
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
			gen_addition(k, x, y, 0, 0);
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
	int i;
	uint64_t a, b, c;

	kann_mopt_init(&mo);
	mo.max_epoch = 20;
	mo.max_rnn_len = bit_len;
	ann = kann_rnn_gen_vanilla(4, 2, 1, 20);
	kann_train(&mo, ann, add_reader, 0);

	x = (float**)calloc(bit_len, sizeof(float*));
	for (i = 0; i < bit_len; ++i)
		x[i] = (float*)calloc(4, sizeof(float));
	gen_addition(0, x, 0, &a, &b);
	y = kann_rnn_apply_seq1(ann, bit_len, x);
	for (i = 0, c = 0; i < bit_len; ++i) {
		int k = y[i*2] > y[i*2+1]? 0 : 1;
		putchar('0' + k);
	}
	putchar('\n');
	for (i = 0, c = 0; i < bit_len; ++i) {
		int k = (a+b)>>i&1;
		putchar('0' + k);
	}
	putchar('\n');
//	printf("%ld + %ld = %ld ?\n", (long)a, (long)b, (long)c);

	kann_delete(ann);
	return 0;
}
