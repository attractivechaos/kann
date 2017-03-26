#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "kann.h"
#include "kann_extra/kann_data.h"

#define const_scalar(x) kann_new_scalar(KAD_CONST, (x))

static kann_t *model_gen(int n_in, int n_hidden, int n_code)
{
	kad_node_t *x, *t, *s, *mu, *sigma;

	// encoder
	x = kad_feed(2, 1, n_in), x->ext_flag |= KANN_F_IN | KANN_F_TRUTH;
	t = kad_tanh(kann_layer_dense(x, n_hidden));
	mu = kann_layer_dense(t, n_code);
	sigma = kad_relu(kann_layer_dense(t, n_code));
	t = kad_add(kad_sample_normal(sigma), mu), t->ext_label = 1;

	// decoder
	t = kad_tanh(kann_layer_dense(t, n_hidden));
	t = kad_sigm(kann_layer_dense(t, n_in)), t->ext_flag = KANN_F_OUT;
	t = kad_ce_bin(t, x);
	t = kad_mul(t, const_scalar((float)n_in));

	// latent loss
	s = kad_add(kad_square(sigma), const_scalar(1e-6f)); // sigma^2, plus a pseudo-count
	s = kad_sub(s, kad_log(s));              // sigma^2 - log(sigma^2)
	s = kad_add(s, kad_square(mu));          // mu^2 + sigma^2 - log(sigma^2)
	s = kad_sub(s, const_scalar(1.0f));      // mu^2 + sigma^2 - log(sigma^2) - 1
	s = kad_reduce_sum(s, 1);
	s = kad_mul(s, const_scalar(0.5f));
	s = kad_reduce_mean(s, 0);

	t = kad_add(t, s);
	t = kad_mul(t, const_scalar(1.0f / (n_in + 2 * n_code))), t->ext_flag |= KANN_F_COST;
	return kann_new(t, 0);
}

int main(int argc, char *argv[])
{
	int max_epoch = 50, mini_size = 64, max_drop_streak = 10;
	int i, j, c, n_hidden = 64, n_code = 2, seed = 11, to_apply = 0, n_gen = 0;
	kann_data_t *in = 0;
	kann_t *ann = 0;
	char *out_fn = 0, *in_fn = 0;
	float lr = 0.01f, frac_val = 0.1f;

	while ((c = getopt(argc, argv, "n:s:r:m:B:o:i:Ag:c:")) >= 0) {
		if (c == 'n') n_hidden = atoi(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 'i') in_fn = optarg;
		else if (c == 'o') out_fn = optarg;
		else if (c == 'r') lr = atof(optarg);
		else if (c == 'm') max_epoch = atoi(optarg);
		else if (c == 'B') mini_size = atoi(optarg);
		else if (c == 'A') to_apply = 1;
		else if (c == 'c') n_code = atoi(optarg);
		else if (c == 'g') n_gen = atoi(optarg);
	}
	if (argc - optind < 1 && in_fn == 0 && n_gen == 0) {
		FILE *fp = stdout;
		fprintf(fp, "Usage: vae [options] <in.knd>\n");
		fprintf(fp, "Options:\n");
		fprintf(fp, "  Model construction:\n");
		fprintf(fp, "    -i FILE     read trained model from FILE []\n");
		fprintf(fp, "    -o FILE     save trained model to FILE []\n");
		fprintf(fp, "    -s INT      random seed [%d]\n", seed);
		fprintf(fp, "    -n INT      number of hidden neurons [%d]\n", n_hidden);
		fprintf(fp, "    -c INT      number of codes [%d]\n", n_code);
		fprintf(fp, "  Model training:\n");
		fprintf(fp, "    -r FLOAT    learning rate [%g]\n", lr);
		fprintf(fp, "    -m INT      max number of epochs [%d]\n", max_epoch);
		fprintf(fp, "    -B INT      mini-batch size [%d]\n", mini_size);
		fprintf(fp, "  Prediction and generation:\n");
		fprintf(fp, "    -A          reconstruct input\n");
		fprintf(fp, "    -g INT      generate INT samples [%d]\n", n_gen);
		return 1;
	}

	kad_trap_fe();
	kann_srand(seed);
	if (argc - optind >= 1)
		in = kann_data_read(argv[optind]);
	if (in_fn) {
		ann = kann_load(in_fn);
		if (in) assert(kann_dim_in(ann) == in->n_col);
	}

	if (!to_apply && n_gen == 0) { // train
		if (!ann)
			ann = model_gen(in->n_col, n_hidden, n_code);
		kann_train_fnn1(ann, lr, mini_size, max_epoch, max_drop_streak, frac_val, in->n_row, in->x, in->x);
		if (out_fn) kann_save(out_fn, ann);
	} else if (to_apply) { // apply
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
	} else {
		kad_node_t *t, *out;
		int j, n_out;
		kann_set_batch_size(ann, 1);
		out = ann->v[kann_find(ann, KANN_F_OUT, 0)];
		t = ann->v[kann_find(ann, 0, 1)];
		n_code = kad_len(t);
		n_out = kad_len(out);
		for (j = 0; j < n_gen; ++j) {
			kad_eval_disable(t);
			for (i = 0; i < n_code; ++i)
				t->x[i] = kad_drand_normal(0);
			kann_eval(ann, KANN_F_OUT, 0);
			printf("%d", j + 1);
			for (i = 0; i < n_out; ++i)
				printf("\t%g", out->x[i] + 1.0f - 1.0f);
			putchar('\n');
		}
	}

	kann_delete(ann);
	if (in) kann_data_free(in);
	return 0;
}
