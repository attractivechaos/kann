#include <math.h>
#include <stdio.h>
#include <float.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include "kann.h"

#define VERSION "r490"

typedef struct {
	int len, n_char, n_para, *para_len;
	uint8_t *data, **para;
	int c2i[256];
} tg_data_t;

#define kv_roundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))

uint8_t *tg_read_file(const char *fn, int *_len)
{
	const int buf_len = 0x10000;
	int len = 0, max = 0, l;
	FILE *fp;
	uint8_t *buf, *s = 0;

	fp = fn && strcmp(fn, "-")? fopen(fn, "rb") : stdin;
	buf = (uint8_t*)malloc(buf_len);
	while ((l = fread(buf, 1, buf_len, fp)) > 0) {
		if (len + l > max) {
			max = len + buf_len;
			kv_roundup32(max);
			s = (uint8_t*)realloc(s, max);
		}
		memcpy(&s[len], buf, l);
		len += l;
	}
	s = (uint8_t*)realloc(s, len);
	*_len = len;
	fclose(fp);
	free(buf);
	return s;
}

tg_data_t *tg_init(const char *fn)
{
	int i, j, st, k;
	tg_data_t *tg;
	tg = (tg_data_t*)calloc(1, sizeof(tg_data_t));
	tg->data = tg_read_file(fn, &tg->len);
	for (i = 0; i < tg->len; ++i)
		tg->c2i[tg->data[i]] = 1;
	for (i = j = 0; i < 256; ++i)
		if (tg->c2i[i] == 0) tg->c2i[i] = -1;
		else tg->c2i[i] = j++;
	tg->n_char = j;
	for (i = 1, st = 0, tg->n_para = 0; i < tg->len; ++i)
		if (tg->data[i] == '\n' && tg->data[i-1] == '\n' && i - st > 1)
			++tg->n_para, st = i + 1;
	if (i - st > 1) ++tg->n_para;
	tg->para = (uint8_t**)calloc(tg->n_para, sizeof(uint8_t*));
	tg->para_len = (int*)calloc(tg->n_para, sizeof(int));
	for (i = 1, st = k = 0; i < tg->len; ++i)
		if (tg->data[i] == '\n' && tg->data[i-1] == '\n' && i - st > 1)
			tg->para[k] = &tg->data[st], tg->para_len[k++] = i - st, st = i + 1;
	if (i - st > 1) tg->para[k] = &tg->data[st], tg->para_len[k++] = i - st;
	for (i = 0; i < tg->len; ++i)
		tg->data[i] = tg->c2i[tg->data[i]];
	return tg;
}

void tg_save(const char *fn, kann_t *ann, const int c2i[256])
{
	FILE *fp;
	fp = fn && strcmp(fn, "-")? fopen(fn, "wb") : stdout;
	kann_save_fp(fp, ann);
	fwrite(c2i, sizeof(int), 256, fp);
	fclose(fp);
}

kann_t *tg_load(const char *fn, int c2i[256])
{
	FILE *fp;
	kann_t *ann;
	fp = fn && strcmp(fn, "-")? fopen(fn, "rb") : stdin;
	ann = kann_load_fp(fp);
	fread(c2i, sizeof(int), 256, fp);
	fclose(fp);
	return ann;
}

void tg_gen(FILE *fp, kann_t *ann, float temp, int len, const int c2i[256], const char *seed)
{
	int i, c, n_char, i2c[256], i_temp;
	float x[256];
	memset(i2c, 0, 256 * sizeof(int));
	for (i = 0; i < 256; ++i)
		if (c2i[i] >= 0) i2c[c2i[i]] = i;
	n_char = kann_dim_in(ann);
	i_temp = kann_find(ann, 0, -1);
	if (i_temp >= 0) ann->v[i_temp]->x[0] = 1.0f / temp;
	kann_rnn_start(ann);
	for (c = 0; c < ann->n; ++c) {
		kad_node_t *p = ann->v[c];
		if (p->pre) {
			int l = kad_len(p);
			for (i = 0; i < l; ++i)
				p->x[i] = 2.0 * kann_drand() - 1.0;
		}
	}
	if (seed) {
		const char *p;
		for (p = seed; *p; ++p) {
			const float *y;
			float max = -1.0f;
			int max_c = -1;
			c = c2i[(int)*p];
			assert(c >= 0);
			memset(x, 0, n_char * sizeof(float));
			x[c] = 1.0f;
			y = kann_apply1(ann, x);
			for (c = 0; c < n_char; ++c)
				if (max < y[c]) max = y[c], max_c = c;
			c = max_c;
		}
		fprintf(fp, "%s%c", seed, i2c[c]);
	} else c = c2i[(int)' '];
	for (i = 0; i < len; ++i) {
		float s, r;
		const float *y;
		memset(x, 0, n_char * sizeof(float));
		x[c] = 1.0f;
		y = kann_apply1(ann, x);
		r = kann_drand();
		for (c = 0, s = 0.0f; c < n_char; ++c)
			if (s + y[c] >= r) break;
			else s += y[c];
		fputc(i2c[c], fp);
	}
	fputc('\n', fp);
	kann_rnn_end(ann);
	if (i_temp >= 0) ann->v[i_temp]->x[0] = 1.0f;
}

float tg_perplexity(kann_t *ann, const tg_data_t *tg)
{
	const float tiny = 1e-6;
	float x[256], p;
	double loss = 0.0;
	int i;
	kann_rnn_start(ann);
	for (i = 0; i < tg->len - 1; ++i) {
		const float *y;
		memset(x, 0, 256 * sizeof(float));
		x[tg->data[i]] = 1.0f;
		y = kann_apply1(ann, x);
		p = y[tg->data[i+1]];
		loss += logf(p > tiny? p : tiny);
	}
	kann_rnn_end(ann);
	return (float)exp(-loss / (tg->len - 1));
}

int tg_urnn_start(kann_t *ann, int batch_size)
{
	int i, j, n, cnt = 0;
	for (i = 0; i < ann->n; ++i) {
		kad_node_t *p = ann->v[i];
		if (p->pre && p->n_d >= 2 && p->pre->n_d == p->n_d && p->pre->n_child == 0 && kad_len(p)/p->d[0] == kad_len(p->pre)/p->pre->d[0])
			p->pre->flag = 0;
	}
	kann_set_batch_size(ann, batch_size);
	for (i = 0; i < ann->n; ++i) {
		kad_node_t *p = ann->v[i];
		if (p->pre && p->n_d >= 2 && p->pre->n_d == p->n_d && p->pre->n_child == 0 && kad_len(p) == kad_len(p->pre)) {
			kad_node_t *q = p->pre;
			n = kad_len(p) / p->d[0];
			memset(p->x, 0, p->d[0] * n * sizeof(float));
			if (q->x)
				for (j = 0; j < p->d[0]; ++j)
					memcpy(&p->x[j * n], q->x, n * sizeof(float));
			q->x = p->x;
			++cnt;
		}
	}
	return cnt;
}

void tg_train(kann_t *ann, const tg_data_t *tg, float lr, int ulen, int vlen, int cs, int mbs, int max_epoch, float grad_clip, const char *fn, int batch_len, int n_threads)
{
	int i, epoch, u, n_var, n_char;
	float **x, **y, *r;
	const uint8_t **p;
	kann_t *ua;

	batch_len = batch_len < tg->len? batch_len : tg->len;
	n_char = kann_dim_in(ann);
	x = (float**)calloc(ulen, sizeof(float*));
	y = (float**)calloc(ulen, sizeof(float*));
	for (u = 0; u < ulen; ++u) {
		x[u] = (float*)calloc(n_char * mbs, sizeof(float));
		y[u] = (float*)calloc(n_char * mbs, sizeof(float));
	}
	n_var = kann_size_var(ann);
	r = (float*)calloc(n_var, sizeof(float));
	p = (const uint8_t**)calloc(mbs, sizeof(const uint8_t*));

	ua = kann_unroll(ann, ulen);
	tg_urnn_start(ua, mbs);
	kann_mt(ua, n_threads, mbs);
	kann_switch(ua, 1);
	kann_feed_bind(ua, KANN_F_IN,  100, x);
	kann_feed_bind(ua, KANN_F_TRUTH, 0, y);
	for (epoch = 0; epoch < max_epoch; ++epoch) {
		double cost = 0.0;
		int c, j, b, tot = 0, ctot = 0, n_cerr = 0;
		for (i = 0; i < batch_len; i += mbs * cs * ulen) {
			for (b = 0; b < mbs; ++b)
				p[b] = tg->data + (int)((tg->len - ulen * cs - 1) * kad_drand(0)) + 1;
			for (j = 0; j < ua->n; ++j) // reset initial hidden values to zero
				if (ua->v[j]->pre)
					memset(ua->v[j]->x, 0, kad_len(ua->v[j]) * sizeof(float));
			for (c = 0; c < cs; ++c) {
				int ce_len = c? ulen : ulen - vlen;
				for (u = 0; u < ulen; ++u) {
					memset(x[u], 0, mbs * n_char * sizeof(float));
					memset(y[u], 0, mbs * n_char * sizeof(float));
				}
				for (b = 0; b < mbs; ++b) {
					for (u = 0; u < ulen; ++u) {
						x[u][b * n_char + p[b][u-1]] = 1.0f;
						if (c || u >= vlen)
							y[u][b * n_char + p[b][u]] = 1.0f;
					}
					p[b] += ulen;
				}
				cost += kann_cost(ua, 0, 1) * ulen * mbs;
				n_cerr += kann_class_error(ua, &b);
				tot += ce_len * mbs, ctot += b;
				if (grad_clip > 0.0f) kann_grad_clip(grad_clip, n_var, ua->g);
				kann_RMSprop(n_var, lr, 0, 0.9f, ua->g, ua->x, r);
			}
		}
		fprintf(stderr, "epoch: %d; running cost: %g (class error: %.2f%%)\n", epoch+1, cost / tot, 100.0 * n_cerr / ctot);
		tg_gen(stderr, ann, 0.4f, 100, tg->c2i, "is");
		if (fn) tg_save(fn, ann, tg->c2i);
	}
	kann_delete_unrolled(ua);

	for (u = 0; u < ulen; ++u) {
		free(x[u]); free(y[u]);
	}
	free(r); free(y); free(x); free(p);
}

static kann_t *model_gen(int model, int n_char, int n_h_layers, int n_h_neurons, float h_dropout, int use_norm)
{
	int i, flag = use_norm? KANN_RNN_NORM : 0;
	kad_node_t *t, *t1;
	t = kann_layer_input(n_char), t->ext_label = 100;
	for (i = 0; i < n_h_layers; ++i) {
		if (model == 0) t = kann_layer_rnn(t, n_h_neurons, flag);
		else if (model == 1) t = kann_layer_lstm(t, n_h_neurons, flag);
		else if (model == 2) t = kann_layer_gru(t, n_h_neurons, flag);
		t = kann_layer_dropout(t, h_dropout);
	}
	t = kann_layer_dense(t, n_char);
	t1 = kann_new_scalar(KAD_CONST, 1.0f), t1->ext_label = -1; // -1 is for backward compatibility
	t = kad_mul(t, t1); // t1 is the inverse of temperature
	t = kad_softmax(t), t->ext_flag |= KANN_F_OUT;
	t1 = kad_feed(2, 1, n_char), t1->ext_flag |= KANN_F_TRUTH;
	t = kad_ce_multi(t, t1), t->ext_flag |= KANN_F_COST;
	return kann_new(t, 0);
}

int main(int argc, char *argv[])
{
	int c, seed = 11, ulen = 70, vlen = 10, n_h_layers = 1, n_h_neurons = 128, model = 2, max_epoch = 50, mbs = 64, c2i[256];
	int len_gen = 1000, use_norm = 1, batch_len = 1000000, n_threads = 1, cal_perp = 0, cs = 100;
	float h_dropout = 0.0f, temp = 0.5f, lr = 0.01f, grad_clip = 10.0f;
	kann_t *ann = 0;
	char *fn_in = 0, *fn_out = 0, *prefix = 0;

	while ((c = getopt(argc, argv, "n:l:s:r:m:B:o:i:d:b:T:M:u:L:g:Np:t:xv:c:")) >= 0) {
		if (c == 'n') n_h_neurons = atoi(optarg);
		else if (c == 'l') n_h_layers = atoi(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 'i') fn_in = optarg;
		else if (c == 'o') fn_out = optarg;
		else if (c == 'r') lr = atof(optarg);
		else if (c == 'm') max_epoch = atoi(optarg);
		else if (c == 'B') mbs = atoi(optarg);
		else if (c == 'd') h_dropout = atof(optarg);
		else if (c == 'T') temp = atof(optarg);
		else if (c == 'c') cs = atoi(optarg);
		else if (c == 'u') ulen = atoi(optarg);
		else if (c == 'v') vlen = atoi(optarg);
		else if (c == 'L') len_gen = atoi(optarg);
		else if (c == 'g') grad_clip = atof(optarg);
		else if (c == 'N') use_norm = 0;
		else if (c == 'p') prefix = optarg;
		else if (c == 'b') batch_len = atoi(optarg);
		else if (c == 't') n_threads = atoi(optarg);
		else if (c == 'x') cal_perp = 1;
		else if (c == 'M') {
			if (strcmp(optarg, "rnn") == 0) model = 0;
			else if (strcmp(optarg, "lstm") == 0) model = 1;
			else if (strcmp(optarg, "gru") == 0) model = 2;
		}
	}
	if (vlen >= ulen) vlen = ulen - 1;
	if (argc == optind && fn_in == 0) {
		FILE *fp = stdout;
		fprintf(fp, "Usage: textgen [options] <in.txt>\n");
		fprintf(fp, "Options:\n");
		fprintf(fp, "  Model construction:\n");
		fprintf(fp, "    -i FILE     read trained model from FILE []\n");
		fprintf(fp, "    -o FILE     save trained model to FILE []\n");
		fprintf(fp, "    -s INT      random seed [%d]\n", seed);
		fprintf(fp, "    -l INT      number of hidden layers [%d]\n", n_h_layers);
		fprintf(fp, "    -n INT      number of hidden neurons per layer [%d]\n", n_h_neurons);
		fprintf(fp, "    -M STR      model: rnn, lstm or gru [gru]\n");
		fprintf(fp, "    -N          don't use layer normalization\n");
		fprintf(fp, "  Model training:\n");
		fprintf(fp, "    -r FLOAT    learning rate [%g]\n", lr);
		fprintf(fp, "    -d FLOAT    dropout at the hidden layer(s) [%g]\n", h_dropout);
		fprintf(fp, "    -m INT      max number of epochs [%d]\n", max_epoch);
		fprintf(fp, "    -B INT      mini-batch size [%d]\n", mbs);
		fprintf(fp, "    -u INT      max unroll [%d]\n", ulen);
		fprintf(fp, "    -v INT      burn-in length [%d]\n", vlen);
		fprintf(fp, "    -g FLOAT    gradient clipping threshold [%g]\n", grad_clip);
		fprintf(fp, "    -c INT      size of a batch [%d]\n", batch_len);
		fprintf(fp, "    -b          use minibatch (run faster but converge slower)\n");
		fprintf(fp, "    -x          compute perplexity at the end\n");
		fprintf(fp, "  Text generation:\n");
		fprintf(fp, "    -p STR      prefix []\n");
		fprintf(fp, "    -T FLOAT    temperature [%g]\n", temp);
		fprintf(fp, "    -L INT      length of text to generate [%d]\n", len_gen);
		return 1;
	}

	fprintf(stderr, "Version: %s\n", VERSION);
	fprintf(stderr, "Command line:");
	for (c = 0; c < argc; ++c)
		fprintf(stderr, " %s", argv[c]);
	fprintf(stderr, "\n");
	kann_srand(seed);
	kad_trap_fe();
	if (fn_in) ann = tg_load(fn_in, c2i);

	if (argc - optind >= 1) { // train
		tg_data_t *tg;
		tg = tg_init(argv[optind]);
		fprintf(stderr, "Read %d paragraphs and %d characters; alphabet size %d\n", tg->n_para, tg->len, tg->n_char);
		if (!ann) ann = model_gen(model, tg->n_char, n_h_layers, n_h_neurons, h_dropout, use_norm);
		tg_train(ann, tg, lr, ulen, vlen, cs, mbs, max_epoch, grad_clip, fn_out, batch_len, n_threads);
		if (cal_perp) fprintf(stderr, "Character-level perplexity: %g\n", tg_perplexity(ann, tg));
		free(tg->data); free(tg);
	} else tg_gen(stdout, ann, temp, len_gen, c2i, prefix);

	kann_delete(ann);
	return 0;
}
