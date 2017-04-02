#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include "kann.h"
#include "dna-io.h"
#include "kseq.h"
KSEQ_DECLARE(gzFile)

static void train(kann_t *ann, dn_seqs_t *tr, dn_seqs_t *va, float lr, int mb_size, int max_epoch, int chunk_size, const char *fn_out)
{
	int epoch, i, j, k, b, len, n_in, n_out, n_var, *cnt;
	kad_node_t *in;
	float *x, *y, *r;
	uint64_t tot_len_tr;

	in = ann->v[kann_find(ann, KANN_F_IN, 0)];
	len = in->d[2];
	n_in  = kann_dim_in(ann);
	n_out = kann_dim_out(ann);
	n_var = kann_size_var(ann);

	x = (float*)calloc(mb_size * n_in,  sizeof(float));
	y = (float*)calloc(mb_size * n_out, sizeof(float));
	r = (float*)calloc(n_var, sizeof(float));
	cnt = (int*)alloca(n_out * sizeof(int));

	for (i = 0, tot_len_tr = 0; i < tr->n; ++i)
		tot_len_tr += tr->len[i];

	kann_feed_bind(ann, KANN_F_IN, 0, &x);
	kann_feed_bind(ann, KANN_F_TRUTH, 0, &y);
	for (epoch = 0; epoch < max_epoch; ++epoch) {
		double cost = 0.0;
		int tot = 0, n_proc = 0;
		kann_set_batch_size(ann, mb_size);
		while (n_proc < chunk_size) {
			for (b = 0; b < mb_size; ++b) {
				double u;
				int partial_len = 0, sum;
				u = kann_drand();
				for (i = 0; i < tr->n; ++i)
					if (partial_len + tr->len[i] > tot_len_tr * u)
						break;
				j = (int)((tr->len[i] - len + 1) * kann_drand());
				dn_seq2vec_ds(len, &tr->seq[i][j], &x[b * n_in]);
				memset(cnt, 0, n_out * sizeof(int));
				for (k = 0, sum = 0; k < len; ++k)
					++cnt[tr->lbl[i][j + k]], ++sum;
				for (k = 0; k < n_out; ++k)
					y[b * n_out + k] = (float)cnt[k] / sum;
			}
			cost += kann_cost(ann, 0, 1) * mb_size;
			tot += mb_size;
			kann_RMSprop(n_var, lr, 0, 0.9f, ann->g, ann->x, r);
			n_proc += len * mb_size;
		}
		fprintf(stderr, "epoch: %d; running cost: %g\n", epoch+1, cost / tot);
		if (fn_out) kann_save(fn_out, ann);
	}

	free(r); free(y); free(x);
}

static kann_t *model_gen(int n_lbl, int len, int k_size, int n_flt, int n_h_neurons)
{
	int i;
	kad_node_t *in, *t[2], *wc, *wf = 0, *bf, *wo, *wc2;

	wc = kann_new_weight_conv1d(n_flt, 4, k_size);
	wc2 = kann_new_weight_conv1d(n_flt, n_flt, 5);
	bf = kann_new_bias(n_h_neurons);
	wo = kann_new_weight(1, n_h_neurons);

	in = kad_feed(3, 1, 8, len), in->ext_flag = KANN_F_IN;
	t[0] = kad_slice(in, 1, 0, 4);
	t[1] = kad_slice(in, 1, 4, 8);
	for (i = 0; i < 2; ++i) {
		t[i] = kad_relu(kad_conv1d(t[i], wc, 1, 0));
		t[i] = kad_max1d(t[i], 3, 3, 0);
		t[i] = kad_relu(kad_conv1d(t[i], wc2, 1, 0));
		t[i] = kad_max1d(t[i], 3, 3, 0);
		if (wf == 0)
			wf = kann_new_weight(n_h_neurons, kad_len(t[i]) / t[i]->d[0]);
		t[i] = kad_relu(kad_add(kad_cmul(t[i], wf), bf));
		t[i] = kad_cmul(t[i], wo);
	}
	return kann_new(kann_layer_cost(kad_add(t[0], t[1]), n_lbl, KANN_C_CEM), 0);
}

int main(int argc, char *argv[])
{
	int c, seed = 11, ws = 100, n_flt = 32, k_size = 17, n_h_neurons = 64, to_apply = 0, max_epoch = 100, mb_size = 64, chunk_size = 50000000, n_threads = 1;
	float h_dropout = 0.1f, lr = 0.001f;
	kann_t *ann = 0;
	char *fn_in = 0, *fn_out = 0;
	dn_seqs_t *s = 0;

	fprintf(stderr, "Command line: ");
	for (c = 0; c < argc; ++c) {
		if (c) fprintf(stderr, " ");
		fprintf(stderr, "%s", argv[c]);
	}
	fputc('\n', stderr);
	while ((c = getopt(argc, argv, "n:s:r:m:B:o:i:d:k:f:w:Ac:t:")) >= 0) {
		if (c == 'n') n_h_neurons = atoi(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 'i') fn_in = optarg;
		else if (c == 'o') fn_out = optarg;
		else if (c == 'f') n_flt = atoi(optarg);
		else if (c == 'r') lr = atof(optarg);
		else if (c == 'm') max_epoch = atoi(optarg);
		else if (c == 'B') mb_size = atoi(optarg);
		else if (c == 'd') h_dropout = atof(optarg);
		else if (c == 'k') k_size = atoi(optarg);
		else if (c == 'A') to_apply = 1;
		else if (c == 't') n_threads = atoi(optarg);
		else if (c == 'w') ws = atoi(optarg);
		else if (c == 'c') {
			char *p;
			chunk_size = strtol(optarg, &p, 10);
			if (*p == 'm' || *p == 'M') chunk_size *= 1024*1024;
			else if (*p == 'k' || *p == 'K') chunk_size *= 1024;
		}
	}
	if (argc == optind && fn_in == 0) {
		FILE *fp = stdout;
		fprintf(fp, "Usage: seq-cnn [options] <in.fq>\n");
		fprintf(fp, "Options:\n");
		fprintf(fp, "  Model construction:\n");
		fprintf(fp, "    -i FILE     read trained model from FILE []\n");
		fprintf(fp, "    -o FILE     save trained model to FILE []\n");
		fprintf(fp, "    -s INT      random seed [%d]\n", seed);
		fprintf(fp, "    -n INT      number of hidden neurons per layer [%d]\n", n_h_neurons);
		fprintf(fp, "    -w INT      window size [%d]\n", ws);
		fprintf(fp, "    -f INT      number of filters at the first layer [%d]\n", n_flt);
		fprintf(fp, "    -k INT      kernel length [%d]\n", k_size);
		fprintf(fp, "  Model training:\n");
		fprintf(fp, "    -r FLOAT    learning rate [%g]\n", lr);
//		fprintf(fp, "    -d FLOAT    dropout at the hidden layer(s) [%g]\n", h_dropout);
		fprintf(fp, "    -m INT      max number of epochs [%d]\n", max_epoch);
		fprintf(fp, "    -c STR      process INT nucleotides per epoch [%d]\n", chunk_size);
		fprintf(fp, "    -B INT      mini-batch size [%d]\n", mb_size);
		return 1;
	}

	kann_srand(seed);
	kad_trap_fe();
	if (fn_in) ann = kann_load(fn_in);
	if (ann == 0 || !to_apply)
		s = dn_read(argv[optind]);

	if (to_apply && ann) {
		gzFile fp;
		kseq_t *seq;
		int ws;
		float *x;
		uint8_t *seq4;

		ws = kann_dim_in(ann) / 8;
		x = (float*)malloc(ws * 8 * sizeof(float));
		seq4 = (uint8_t*)calloc(ws + 1, 1);
		fp = gzopen(argv[optind], "r");
		seq = kseq_init(fp);
		while (kseq_read(seq) >= 0) {
			int i, j;
			for (i = 0; i < seq->seq.l; i += ws) {
				const float *y;
				float true_frac = -1.0f;
				for (j = i; j < i + ws; ++j)
					seq4[j-i] = seq_nt4_table[(int)seq->seq.s[j]];
				dn_seq2vec_ds(ws, seq4, x);
				y = kann_apply1(ann, x);
				if (seq->qual.l) {
					int n_sig = 0;
					for (j = i; j < i + ws; ++j) {
						int sig = seq->qual.s[j] - 33;
						if (sig) ++n_sig;
						seq->seq.s[j] = sig? toupper(seq->seq.s[j]) : tolower(seq->seq.s[j]);
					}
					true_frac = (float)n_sig / ws;
				}
				memcpy(seq4, &seq->seq.s[i], ws);
				printf("%s\t%.2f\t%.2f\n", (char*)seq4, 100.0 * true_frac, 100.0 * y[1]);
			}
		}
		kseq_destroy(seq);
		gzclose(fp);
		free(seq4); free(x);
	} else { // train
		if (ann == 0) ann = model_gen(s->n_lbl, ws, k_size, n_flt, n_h_neurons);
		if (n_threads > 1) kann_mt(ann, n_threads, mb_size);
		//kad_print_graph(stderr, ann->n, ann->v);
		train(ann, s, 0, lr, mb_size, max_epoch, chunk_size, fn_out);
	}

	kann_delete(ann);
	if (s) dn_destroy(s);
	return 0;
}
