#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "kann.h"
#include "kann_extra/kann_data.h"

static kann_t *model_gen(int n_in, int n_hidden, float i_dropout)
{
	kad_node_t *x, *t, *w, *r;
	w = kann_new_weight(n_hidden, n_in);
	r = kann_new_scalar(KAD_VAR, sqrtf((float)n_in / n_hidden));
	x = kad_feed(2, 1, n_in), x->ext_flag |= KANN_F_IN | KANN_F_TRUTH;
	t = kann_layer_dropout(x, i_dropout);
	t = kad_tanh(kad_add(kad_cmul(t, w), kann_new_bias(n_hidden)));
	t = kad_mul(t, r);
	t = kad_add(kad_matmul(t, w), kann_new_bias(n_in));
	t = kad_sigm(t), t->ext_flag = KANN_F_OUT;
	t = kad_ce_bin(t, x), t->ext_flag = KANN_F_COST;
	return kann_new(t, 0);
}

int main(int argc, char *argv[])
{
	int max_epoch = 50, mini_size = 64, max_drop_streak = 10;
	int i, j, c, n_hidden = 64, seed = 11, to_apply = 0;
	kann_data_t *in = 0;
	kann_t *ann = 0;
	char *out_fn = 0, *in_fn = 0;
	float lr = 0.01f, frac_val = 0.1f, i_dropout = 0.0f;

	while ((c = getopt(argc, argv, "n:s:r:m:B:o:i:d:A")) >= 0) {
		if (c == 'n') n_hidden = atoi(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 'i') in_fn = optarg;
		else if (c == 'o') out_fn = optarg;
		else if (c == 'r') lr = atof(optarg);
		else if (c == 'm') max_epoch = atoi(optarg);
		else if (c == 'B') mini_size = atoi(optarg);
		else if (c == 'd') i_dropout = atof(optarg);
		else if (c == 'A') to_apply = 1;
	}
	if (argc - optind < 1) {
		FILE *fp = stdout;
		fprintf(fp, "Usage: ae [options] <in.knd>\n");
		fprintf(fp, "Options:\n");
		fprintf(fp, "  Model construction:\n");
		fprintf(fp, "    -i FILE     read trained model from FILE []\n");
		fprintf(fp, "    -o FILE     save trained model to FILE []\n");
		fprintf(fp, "    -s INT      random seed [%d]\n", seed);
		fprintf(fp, "    -n INT      number of hidden neurons [%d]\n", n_hidden);
		fprintf(fp, "    -d FLOAT    dropout at the input layer [%g]\n", i_dropout);
		fprintf(fp, "  Model training:\n");
		fprintf(fp, "    -r FLOAT    learning rate [%g]\n", lr);
		fprintf(fp, "    -m INT      max number of epochs [%d]\n", max_epoch);
		fprintf(fp, "    -B INT      mini-batch size [%d]\n", mini_size);
		return 1;
	}

	kad_trap_fe();
	kann_srand(seed);
	in = kann_data_read(argv[optind]);
	if (in_fn) {
		ann = kann_load(in_fn);
		assert(kann_dim_in(ann) == in->n_col);
	}

	if (!to_apply) { // train
		if (!ann)
			ann = model_gen(in->n_col, n_hidden, i_dropout);
		kann_train_fnn1(ann, lr, mini_size, max_epoch, max_drop_streak, frac_val, in->n_row, in->x, in->x);
		if (out_fn) kann_save(out_fn, ann);
	} else { // apply
		kann_switch(ann, 0);
		for (i = 0; i < in->n_row; ++i) {
			const float *y;
			y = kann_apply1(ann, in->x[i]);
			if (in->rname) printf("%s\t", in->rname[i]);
			for (j = 0; j < in->n_col; ++j) {
				if (j) putchar('\t');
				printf("%.3g", y[j] + 1.0f - 1.0f);
			}
			putchar('\n');
		}
	}

	kann_delete(ann);
	kann_data_free(in);
	return 0;
}
