#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include "kann_extra/kann_data.h"
#include "kann.h"

int main(int argc, char *argv[])
{
	kann_t *ann;
	kann_data_t *x, *y;
	kann_mopt_t mo;
	char *fn_in = 0, *fn_out = 0;
	int c, seed = 131, n_h_fc = 128, n_h_flt = 32;
	float dropout = 0.2f;

	kann_mopt_init(&mo);
	while ((c = getopt(argc, argv, "i:o:m:h:f:d:s:")) >= 0) {
		if (c == 'i') fn_in = optarg;
		else if (c == 'o') fn_out = optarg;
		else if (c == 'm') mo.max_epoch = atoi(optarg);
		else if (c == 'h') n_h_fc = atoi(optarg);
		else if (c == 'f') n_h_flt = atoi(optarg);
		else if (c == 'd') dropout = atof(optarg);
		else if (c == 's') seed = atoi(optarg);
	}

	if (argc - optind == 0 || (argc - optind == 1 && fn_in == 0)) {
		FILE *fp = stdout;
		fprintf(fp, "Usage: mnist-cnn [-i model] [-o model] <x.knd> [y.knd]\n");
		return 1;
	}

	kad_trap_fe();
	kann_srand(seed);
	if (fn_in) {
		ann = kann_read(fn_in);
	} else {
		kad_node_t *t;
		t = kad_par(0, 4, 1, 1, 28, 28), t->label = KANN_L_IN;
		t = kad_relu(kann_layer_conv2d(t, n_h_flt, 3, 3, 1, KAD_PAD_AUTO));
		t = kad_relu(kann_layer_conv2d(t, n_h_flt, 3, 3, 1, KAD_PAD_AUTO));
		t = kad_max2d(t, 2, 2, 2, 2, KAD_PAD_AUTO, KAD_PAD_AUTO);
		t = kann_layer_dropout(t, dropout);
		t = kann_layer_linear(t, n_h_fc);
		t = kad_relu(t);
		t = kann_layer_dropout(t, dropout);
		ann = kann_layer_final(t, 10, KANN_C_CEB);
	}

	x = kann_data_read(argv[optind]);
	assert(x->n_col == 28 * 28);
	y = argc - optind >= 2? kann_data_read(argv[optind+1]) : 0;

	if (y) { // training
		assert(y->n_col == 10);
		kann_fnn_train(&mo, ann, x->n_row, x->x, y->x);
		if (fn_out) kann_write(fn_out, ann);
		kann_data_free(y);
	} else { // applying
		int i, j, n_out;
		kann_set_hyper(ann, KANN_H_DROPOUT, 0.0f);
		n_out = kann_n_out(ann);
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
