#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "kann.h"
#include "kann_rand.h"

static int bit_len = 31, n_train = 10000, n_validate = 1000, n_proc_t, n_proc_v;

int add_reader(void *data, int action, int *len, int max_bs, float **x, float **y)
{
	if (action == KANN_RA_RESET) {
		n_proc_t = n_proc_v = 0;
	} else if (action == KANN_RA_READ_TRAIN || action == KANN_RA_READ_VALIDATE) {
		int i, j;
		uint64_t mask = (1ULL<<bit_len) - 1, *a, *b, *c;
		if (action == KANN_RA_READ_TRAIN    && n_proc_t + max_bs > n_train)    return 0;
		if (action == KANN_RA_READ_VALIDATE && n_proc_v + max_bs > n_validate) return 0;
		a = (uint64_t*)alloca(max_bs * 8);
		b = (uint64_t*)alloca(max_bs * 8);
		c = (uint64_t*)alloca(max_bs * 8);
		for (i = 0; i < max_bs; ++i) {
			a[i] = kann_rand() & mask;
			b[i] = kann_rand() & mask;
			c[i] = a[i] + b[i];
		}
		for (j = 0; j < bit_len; ++j) {
			uint64_t z = 1ULL << j;
			memset(x[j], 0, max_bs * 4 * sizeof(float));
			memset(y[j], 0, max_bs * 2 * sizeof(float));
			for (i = 0; i < max_bs; ++i) {
				x[j][a[i]&z] = 1.0f;
				x[j][(b[i]&z)+2] = 1.0f;
				y[j][c[i]&z] = 1.0f;
			}
		}
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

	kann_mopt_init(&mo);
	ann = kann_rnn_gen_vanilla(bit_len * 4, bit_len * 2, 1, 20);
	kann_train(&mo, ann, add_reader, 0);
	kann_delete(ann);
	return 0;
}
