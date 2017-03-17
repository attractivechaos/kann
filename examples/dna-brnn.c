#include <zlib.h>
#include <ctype.h>
#include "kann.h"
#include "kann_extra/kseq.h"
KSTREAM_INIT(gzFile, gzread, 65536)

typedef struct {
	kstring_t s;
} dna_rnn_t;

unsigned char seq_nt4_table[256] = {
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4 /*'-'*/, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};

static inline int kputsn(const char *p, int l, kstring_t *s)
{
	if (s->l + l + 1 >= s->m) {
		char *tmp;
		s->m = s->l + l + 2;
		kroundup32(s->m);
		if ((tmp = (char*)realloc(s->s, s->m)))
			s->s = tmp;
		else
			return EOF;
	}
	memcpy(s->s + s->l, p, l);
	s->l += l;
	s->s[s->l] = 0;
	return l;
}

dna_rnn_t *dr_read(const char *fn)
{
	gzFile fp;
	kstream_t *ks;
	kstring_t str = {0,0,0};
	dna_rnn_t *dr;

	fp = fn && strcmp(fn, "-")? gzopen(fn, "r") : gzdopen(fileno(stdin), "r");
	if (fp == 0) return 0;
	dr = (dna_rnn_t*)calloc(1, sizeof(dna_rnn_t));
	ks = ks_init(fp);
	while (ks_getuntil(ks, KS_SEP_LINE, &str, 0) >= 0)
		kputsn(str.s, str.l, &dr->s);
	free(str.s);
	ks_destroy(ks);
	gzclose(fp);
	return dr;
}

kann_t *dr_model_gen(int n_layer, int n_neuron, float h_dropout)
{
	kad_node_t *s[2], *t, *w, *b, *y;
	int i, k;
	for (k = 0; k < 2; ++k) {
		s[k] = kad_feed(2, 1, 4), s[k]->ext_flag = KANN_F_IN, s[k]->ext_label = k + 1;
		for (i = 0; i < n_layer; ++i) {
			s[k] = kann_layer_gru(s[k], n_neuron, KANN_RNN_NORM);
			if (h_dropout > 0.0f) s[k] = kann_layer_dropout(s[k], h_dropout);
		}
		s[k] = kad_stack(1, &s[k]);
	}
	s[1] = kad_reverse(s[1], 0);
	t = kad_concat(2, 2, s[0], s[1]), w = kann_new_weight(2, n_neuron * 2);
//	t = kad_avg(2, s), w= kann_new_weight(2, n_neuron);
	b = kann_new_bias(2);
	t = kad_softmax(kad_add(kad_cmul(t, w), b));
	y = kad_feed(2, 1, 2), y->ext_flag = KANN_F_TRUTH;
	y = kad_stack(1, &y);
	t = kad_ce_multi(t, y), t->ext_flag = KANN_F_COST;
	return kann_new(t, 0);
}

void dr_train(kann_t *ann, dna_rnn_t *dr, int ulen, float lr, int m_epoch, int mbs, int n_threads, int batch_len, const char *fn)
{
	float **x[2], **y, *r, grad_clip = 10.0f;
	kann_t *ua;
	uint8_t *rev;
	int epoch, u, n_var;

	rev = (uint8_t*)calloc(ulen, 1);
	x[0] = (float**)calloc(ulen, sizeof(float*));
	x[1] = (float**)calloc(ulen, sizeof(float*));
	y    = (float**)calloc(ulen, sizeof(float*));
	for (u = 0; u < ulen; ++u) {
		x[0][u] = (float*)calloc(4 * mbs, sizeof(float));
		x[1][u] = (float*)calloc(4 * mbs, sizeof(float));
		y[u]    = (float*)calloc(2 * mbs, sizeof(float));
	}
	n_var = kann_size_var(ann);
	r = (float*)calloc(n_var, sizeof(float));

	ua = kann_unroll(ann, ulen, ulen, ulen);
	kann_mt(ua, n_threads, mbs);
	kann_switch(ua, 1);
	kann_feed_bind(ua, KANN_F_IN,    1, x[0]);
	kann_feed_bind(ua, KANN_F_IN,    2, x[1]);
	kann_feed_bind(ua, KANN_F_TRUTH, 0, y);
	for (epoch = 0; epoch < m_epoch; ++epoch) {
		double cost = 0.0;
		int i, b, tot = 0, ctot = 0, n_cerr = 0;
		for (i = 0; i < batch_len; i += mbs * ulen) {
			for (u = 0; u < ulen; ++u) {
				memset(x[0][u], 0, 4 * mbs * sizeof(float));
				memset(x[1][u], 0, 4 * mbs * sizeof(float));
				memset(y[u],    0, 2 * mbs * sizeof(float));
			}
			for (b = 0; b < mbs; ++b) {
				unsigned j = (unsigned)((dr->s.l - ulen) * kad_drand(0));
				for (u = 0; u < ulen; ++u) {
					int c = (uint8_t)dr->s.s[j + u];
					int a = isupper(c);
					c = seq_nt4_table[c];
					if (c >= 4) continue;
					x[0][u][b * 4 + c] = 1.0f;
					x[1][ulen - 1 - u][b * 4 + (3 - c)] = 1.0f;
					y[u][b * 2 + a] = 1.0f;
				}
			}
			cost += kann_cost(ua, 0, 1) * ulen * mbs;
			n_cerr += kann_class_error(ua, &b);
			tot += ulen * mbs, ctot += b;
			if (grad_clip > 0.0f) kann_grad_clip(grad_clip, n_var, ua->g);
			kann_RMSprop(n_var, lr, 0, 0.9f, ua->g, ua->x, r);
		}
		fprintf(stderr, "epoch: %d; running cost: %g (class error: %.2f%%)\n", epoch+1, cost / tot, 100.0 * n_cerr / ctot);
		if (fn) kann_save(fn, ann);
	}
	kann_delete_unrolled(ua);

	for (u = 0; u < ulen; ++u) {
		free(x[0][u]); free(x[1][u]); free(y[u]);
	}
	free(r); free(y); free(x[0]); free(x[1]);
}

int main(int argc, char *argv[])
{
	kann_t *ann = 0;
	dna_rnn_t *dr;
	int c, n_layer = 1, n_neuron = 128, ulen = 100;
	int batch_len = 10000000, mbs = 64, m_epoch = 50, n_threads = 1;
	float h_dropout = 0.0f, lr = 0.001f;
	char *fn_out = 0;

	while ((c = getopt(argc, argv, "u:l:n:m:B:o:")) >= 0) {
		if (c == 'u') ulen = atoi(optarg);
		else if (c == 'l') n_layer = atoi(optarg);
		else if (c == 'n') n_neuron = atoi(optarg);
		else if (c == 'r') lr = atof(optarg);
		else if (c == 'm') m_epoch = atoi(optarg);
		else if (c == 'B') mbs = atoi(optarg);
		else if (c == 'o') fn_out = optarg;
	}

	if (argc - optind < 1) {
		fprintf(stderr, "Usage: dna-brnn [options] <seq.txt>\n");
		return 1;
	}

	dr = dr_read(argv[optind]);
	ann = dr_model_gen(n_layer, n_neuron, h_dropout);
	dr_train(ann, dr, ulen, lr, m_epoch, mbs, n_threads, batch_len, fn_out);
	return 0;
}
