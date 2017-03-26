#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include "kann_extra/kann_data.h"
#include "kann.h"

int main(int argc, char *argv[])
{
	kann_t *ann;
	kann_data_t *x, *y;
	char *fn_in = 0, *fn_out = 0;
	int c, mini_size = 64, max_epoch = 20, max_drop_streak = 10, seed = 131, n_h_fc = 128, n_h_flt = 32, n_threads = 1;
	float lr = 0.001f, dropout = 0.2f, frac_val = 0.1f;

	while ((c = getopt(argc, argv, "i:o:m:h:f:d:s:t:v:")) >= 0) {
		if (c == 'i') fn_in = optarg;
		else if (c == 'o') fn_out = optarg;
		else if (c == 'm') max_epoch = atoi(optarg);
		else if (c == 'h') n_h_fc = atoi(optarg);
		else if (c == 'f') n_h_flt = atoi(optarg);
		else if (c == 'd') dropout = atof(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 't') n_threads = atoi(optarg);
		else if (c == 'v') frac_val = atof(optarg);
	}

	if (argc - optind == 0 || (argc - optind == 1 && fn_in == 0)) {
		FILE *fp = stdout;
		fprintf(fp, "Usage: mnist-cnn [-i model] [-o model] [-t nThreads] <x.knd> [y.knd]\n");
		return 1;
	}

	kad_trap_fe();
	kann_srand(seed);
	if (fn_in) {
		ann = kann_load(fn_in);
	} else {
		kad_node_t *t;
		t = kad_feed(4, 1, 1, 28, 28), t->ext_flag |= KANN_F_IN;
		t = kad_relu(kann_layer_conv2d(t, n_h_flt, 3, 3, 1, 1, 0, 0)); // 3x3 kernel; 1x1 stride; 0x0 padding
		t = kad_relu(kann_layer_conv2d(t, n_h_flt, 3, 3, 1, 1, 0, 0));
		t = kad_max2d(t, 2, 2, 2, 2, 0, 0); // 2x2 kernel; 2x2 stride; 0x0 padding
		t = kann_layer_dropout(t, dropout);
		t = kann_layer_dense(t, n_h_fc);
		t = kad_relu(t);
		t = kann_layer_dropout(t, dropout);
		ann = kann_new(kann_layer_cost(t, 10, KANN_C_CEB), 0);
	}

	x = kann_data_read(argv[optind]);
	assert(x->n_col == 28 * 28);
	y = argc - optind >= 2? kann_data_read(argv[optind+1]) : 0;

	if (y) { // training
		assert(y->n_col == 10);
		if (n_threads > 1) kann_mt(ann, n_threads, mini_size);
		kann_train_fnn1(ann, lr, mini_size, max_epoch, max_drop_streak, frac_val, x->n_row, x->x, y->x);
		if (fn_out) kann_save(fn_out, ann);
		kann_data_free(y);
	} else { // applying
		int i, j, n_out;
		kann_switch(ann, 0);
		n_out = kann_dim_out(ann);
		assert(n_out == 10);
		for (i = 0; i < x->n_row; ++i) {
			const float *y;
			y = kann_apply1(ann, x->x[i]);
			if (x->rname) printf("%s\t", x->rname[i]);
			for (j = 0; j < n_out; ++j) {
				if (j) putchar('\t');
				printf("%.3g", y[j] + 1.0f - 1.0f);
			}
			putchar('\n');
		}
	}

	kann_data_free(x);
	kann_delete(ann);
	return 0;
}
