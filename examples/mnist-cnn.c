#include <unistd.h>
#include <assert.h>
#include "models/kann_data.h"
#include "kann.h"

int main(int argc, char *argv[])
{
	kann_t *ann;
	kann_data_t *x, *y;
	kann_mopt_t mo;
	char *fn_in = 0, *fn_out = 0;
	int c;

	kann_mopt_init(&mo);
	while ((c = getopt(argc, argv, "i:o:")) >= 0) {
		if (c == 'i') fn_in = optarg;
		else if (c == 'o') fn_out = optarg;
	}

	if (argc - optind == 0 || (argc - optind == 1 && fn_in == 0)) {
		FILE *fp = stdout;
		fprintf(fp, "Usage: mnist-cnn [-i model] [-o model] <x.knd> [y.knd]\n");
		return 1;
	}

	if (fn_in) {
		ann = kann_read(fn_in);
	} else {
		kad_node_t *t;
		t = kad_par(0, 4, 1, 1, 28, 28), t->label = KANN_L_IN;
		t = kann_layer_conv2d(t, 32, 3, 3, 1, 0);
		t = kad_relu(t);
		t = kann_layer_conv2d(t, 32, 3, 3, 1, 0);
		t = kad_relu(t);
		t = kann_layer_max2d(t, 2, 2, 2, 0);
		t = kann_layer_dropout(t, 0.25f);
		t = kann_layer_linear(t, 128);
		t = kad_relu(t);
		t = kann_layer_dropout(t, 0.5f);
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
	}

	kann_data_free(x);
	kann_delete(ann);
	return 0;
}
