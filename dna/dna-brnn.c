#include <zlib.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include "dna-io.h"
#include "kann.h"

kann_t *dn_model_gen(int n_lbl, int n_layer, int n_neuron, float h_dropout, int is_tied)
{
	kad_node_t *s[2], *t, *w, *b, *y, *par[256]; // for very unreasonably deep models, this may overflow
	int i, k, offset;
	memset(par, 0, sizeof(kad_node_p) * 256);
	for (k = 0; k < 2; ++k) {
		s[k] = kad_feed(2, 1, 4), s[k]->ext_flag = KANN_F_IN, s[k]->ext_label = k + 1;
		offset = 0;
		for (i = 0; i < n_layer; ++i) {
			if (is_tied) {
				kad_node_t *h0;
				h0 = kann_new_leaf2(&offset, par, KAD_CONST, 0.0f, 2, 1, n_neuron);
				s[k] = kann_layer_gru2(&offset, par, s[k], h0, KANN_RNN_NORM);
			} else s[k] = kann_layer_gru(s[k], n_neuron, KANN_RNN_NORM);
			if (h_dropout > 0.0f) s[k] = kann_layer_dropout(s[k], h_dropout);
		}
		s[k] = kad_stack(1, &s[k]); // first and second pivot
	}
	s[1] = kad_reverse(s[1], 0);
	t = kad_concat(2, 2, s[0], s[1]), w = kann_new_weight(n_lbl, n_neuron * 2);
//	t = kad_avg(2, s), w= kann_new_weight(2, n_neuron);
	b = kann_new_bias(n_lbl);
	t = kad_softmax(kad_add(kad_cmul(t, w), b)), t->ext_flag = KANN_F_OUT;
	y = kad_feed(2, 1, n_lbl), y->ext_flag = KANN_F_TRUTH;
	y = kad_stack(1, &y); // third pivot
	t = kad_ce_multi(t, y), t->ext_flag = KANN_F_COST;
	return kann_new(t, 0);
}

void dn_train(kann_t *ann, dn_seqs_t *dr, int ulen, float lr, int m_epoch, int mbs, int n_threads, int batch_len, const char *fn)
{
	float **x[2], **y, *r, grad_clip = 10.0f, min_cost = 1e30f;
	kann_t *ua;
	int epoch, u, k, n_var, tot_len = 0;

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
	for (k = 0; k < dr->n; ++k)
		if (dr->len[k] >= ulen) tot_len += dr->len[k];

	ua = kann_unroll(ann, ulen, ulen, ulen);
	kann_mt(ua, n_threads, mbs);
	kann_set_batch_size(ua, mbs);
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
				int sub, thres, j;
				thres = (int)(kad_drand(0) * tot_len);
				for (k = sub = 0; k < dr->n; ++k) {
					if (dr->len[k] < ulen) continue;
					sub += dr->len[k];
					if (sub >= thres) break;
				}
				j = (int)((dr->len[k] - ulen) * kad_drand(0));
				for (u = 0; u < ulen; ++u) {
					int c = (uint8_t)dr->seq[k][j + u];
					int a = dr->lbl[k][j + u];
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
		if (fn && cost / tot < min_cost) kann_save(fn, ann);
		if (cost / tot < min_cost) min_cost = cost / tot;
	}
	kann_delete_unrolled(ua);

	for (u = 0; u < ulen; ++u) { free(x[0][u]); free(x[1][u]); free(y[u]); }
	free(r); free(y); free(x[0]); free(x[1]);
}

void dn_predict1(kann_t *ua, float **x[2], char *str, int cnt[4])
{
	int u, ulen;
	kad_node_t *out;
	out = ua->v[kann_find(ua, KANN_F_OUT, 0)];
	ulen = out->d[0];
	for (u = 0; u < ulen; ++u) {
		int c = (uint8_t)str[u];
		c = seq_nt4_table[c];
		memset(x[0][u], 0, 4 * sizeof(float));
		memset(x[1][ulen - 1 - u], 0, 4 * sizeof(float));
		if (c >= 4) continue;
		x[0][u][c] = 1.0f;
		x[1][ulen - 1 - u][3 - c] = 1.0f;
	}
	kann_eval(ua, KANN_F_OUT, 0);
	cnt[0] = cnt[1] = cnt[2] = cnt[3] = 0;
	for (u = 0; u < ulen; ++u) {
		float *y = &out->x[u * 2];
		int c = y[0] > y[1]? 0 : 1;
		++cnt[c];
		if (isupper(str[u]) && c == 0) ++cnt[3];
		else if (islower(str[u]) && c == 1) ++cnt[2];
		str[u] = c == 0? tolower(str[u]) : toupper(str[u]);
	}
}

void dn_predict(kann_t *ann, int ulen, char *str)
{
	float **x[2];
	kann_t *ua;
	int i, u, len;
	char *buf;

	buf = (char*)calloc(ulen + 1, 1);
	x[0] = (float**)calloc(ulen, sizeof(float*));
	x[1] = (float**)calloc(ulen, sizeof(float*));
	for (u = 0; u < ulen; ++u) {
		x[0][u] = (float*)calloc(4, sizeof(float));
		x[1][u] = (float*)calloc(4, sizeof(float));
	}

	ua = kann_unroll(ann, ulen, ulen, ulen);
	kann_set_batch_size(ua, 1);
	kann_feed_bind(ua, KANN_F_IN, 1, x[0]);
	kann_feed_bind(ua, KANN_F_IN, 2, x[1]);
	len = strlen(str);
	for (i = 0; i + ulen <= len; i += ulen/2) {
		int cnt[4];
		strncpy(buf, &str[i], ulen);
		dn_predict1(ua, x, buf, cnt);
		printf("%d\t%d\t%s\t%d\t%d\t%d\t%d\n", i, i+ulen, buf, cnt[0], cnt[1], cnt[2], cnt[3]);
	}
	kann_delete_unrolled(ua);

	for (u = 0; u < ulen; ++u) { free(x[0][u]); free(x[1][u]); }
	free(x[0]); free(x[1]); free(buf);
}

int main(int argc, char *argv[])
{
	kann_t *ann = 0;
	dn_seqs_t *dr;
	int c, n_layer = 1, n_neuron = 128, ulen = 100, to_apply = 0;
	int batch_len = 1000000, mbs = 64, m_epoch = 50, n_threads = 1, is_tied = 1;
	float h_dropout = 0.0f, lr = 0.001f;
	char *fn_out = 0, *fn_in = 0;

	while ((c = getopt(argc, argv, "Au:l:n:m:B:o:i:t:Tb:")) >= 0) {
		if (c == 'u') ulen = atoi(optarg);
		else if (c == 'l') n_layer = atoi(optarg);
		else if (c == 'n') n_neuron = atoi(optarg);
		else if (c == 'r') lr = atof(optarg);
		else if (c == 'm') m_epoch = atoi(optarg);
		else if (c == 'B') mbs = atoi(optarg);
		else if (c == 'o') fn_out = optarg;
		else if (c == 'i') fn_in = optarg;
		else if (c == 'A') to_apply = 1;
		else if (c == 't') n_threads = atoi(optarg);
		else if (c == 'T') is_tied = 0;
		else if (c == 'b') batch_len = atoi(optarg);
	}
	if (argc - optind < 1) {
		fprintf(stderr, "Usage: dna-brnn [options] <seq.txt>\n");
		return 1;
	}

	dr = dn_read(argv[optind]);
	if (fn_in) ann = kann_load(fn_in);
	if (!to_apply) {
		if (ann == 0) ann = dn_model_gen(dr->n_lbl, n_layer, n_neuron, h_dropout, is_tied);
		dn_train(ann, dr, ulen, lr, m_epoch, mbs, n_threads, batch_len, fn_out);
	} else if (ann) {
//		dn_predict(ann, ulen, dr->s.s);
	}
	return 0;
}
