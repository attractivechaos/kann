#include <math.h>
#include <stdio.h>
#include <float.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include "kann.h"

#define VERSION "r461"

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

void tg_train(kann_t *ann, const tg_data_t *tg, float lr, int ulen, int mbs, int max_epoch, float grad_clip, const char *fn, int batch_len, int use_mini, int use_para)
{
	int i, epoch, k, n_var, n_char, real_mbs = use_mini? mbs : 1;
	float **x, **y, *r, *g;
	kann_t *ua;

	n_char = kann_dim_in(ann);
	x = (float**)calloc(ulen, sizeof(float*));
	y = (float**)calloc(ulen, sizeof(float*));
	for (k = 0; k < ulen; ++k) {
		x[k] = (float*)calloc(n_char * real_mbs, sizeof(float));
		y[k] = (float*)calloc(n_char * real_mbs, sizeof(float));
	}
	n_var = kann_size_var(ann);
	r = (float*)calloc(n_var, sizeof(float));
	g = (float*)calloc(n_var, sizeof(float));

	ua = kann_unroll(ann, ulen);
	kann_switch(ua, 1);
	kann_set_batch_size(ua, real_mbs);
	kann_feed_bind(ua, KANN_F_IN,    0, x);
	kann_feed_bind(ua, KANN_F_TRUTH, 0, y);
	for (epoch = 0; epoch < max_epoch; ++epoch) {
		double cost = 0.0;
		int j, b, tot = 0, n_cerr = 0, n_batches;
		n_batches = (batch_len <= 0? tg->len : batch_len) / (ulen * mbs) + 1;
		for (i = 0; i < n_batches; ++i) {
			if (use_mini) {
				for (k = 0; k < ulen; ++k) {
					memset(x[k], 0, mbs * n_char * sizeof(float));
					memset(y[k], 0, mbs * n_char * sizeof(float));
				}
				for (b = 0; b < mbs; ++b) {
					if (use_para) {
						k = (int)(tg->n_para * kad_drand(0));
						if (tg->para_len[k] < ulen + 1) {
							--b;
							continue;
						}
						j = (int)((tg->para_len[k] - ulen - 1) * kad_drand(0)) + 1 + (tg->para[k] - tg->data);
					} else j = (int)((tg->len - ulen - 1) * kad_drand(0)) + 1; // randomly draw a position
					for (k = 0; k < ulen; ++k) {
						x[k][b * n_char + tg->data[j + k - 1]] = 1.0f;
						y[k][b * n_char + tg->data[j + k]] = 1.0f;
					}
				}
				cost += kann_cost(ua, 0, 1) * ulen * mbs;
				n_cerr += kann_class_error(ua);
				tot += ulen * mbs;
				if (grad_clip > 0.0f) kann_grad_clip(grad_clip, n_var, ua->g);
				kann_RMSprop(n_var, lr, 0, 0.9f, ua->g, ua->x, r);
			} else {
				j = (int)((tg->len - ulen * mbs - 1) * kad_drand(0)) + 1; // randomly draw a position
				memset(g, 0, n_var * sizeof(float));
				for (b = 0; b < mbs; ++b) { // a mini-batch
					for (k = 0; k < ulen; ++k) {
						memset(x[k], 0, n_char * sizeof(float));
						memset(y[k], 0, n_char * sizeof(float));
						x[k][tg->data[j + b * ulen + k - 1]] = 1.0f;
						y[k][tg->data[j + b * ulen + k]] = 1.0f;
					}
					cost += kann_cost(ua, 0, 1) * ulen;
					n_cerr += kann_class_error(ua);
					tot += ulen;
					for (k = 0; k < n_var; ++k) g[k] += ua->g[k];
					for (k = 0; k < ua->n; ++k) // keep the cycle rolling
						if (ua->v[k]->pre)
							memcpy(ua->v[k]->pre->x, ua->v[k]->x, kad_len(ua->v[k]) * sizeof(float));
				}
				for (k = 0; k < n_var; ++k) g[k] /= mbs;
				if (grad_clip > 0.0f) kann_grad_clip(grad_clip, n_var, g);
				kann_RMSprop(n_var, lr, 0, 0.9f, g, ua->x, r);
				for (k = 0; k < ann->n; ++k) // reset initial hidden values to zero
					if (ann->v[k]->pre)
						memset(ann->v[k]->pre->x, 0, kad_len(ann->v[k]->pre) * sizeof(float));
			}
		}
		fprintf(stderr, "epoch: %d; running cost: %g (class error: %.2f%%)\n", epoch+1, cost / tot, 100.0 * n_cerr / tot);
		tg_gen(stderr, ann, 0.4f, 100, tg->c2i, "is");
		if (fn) tg_save(fn, ann, tg->c2i);
	}
	kann_delete_unrolled(ua);
	fprintf(stderr, "Character-level perplexity: %g\n", tg_perplexity(ann, tg));

	for (k = 0; k < ulen; ++k) {
		free(x[k]);
		free(y[k]);
	}
	free(g); free(r); free(y); free(x);
}

static kann_t *model_gen(int model, int n_char, int n_h_layers, int n_h_neurons, float h_dropout, int use_norm)
{
	int i, flag = use_norm? KANN_RNN_NORM : 0;
	kad_node_t *t, *t1;
	t = kann_layer_input(n_char);
	for (i = 0; i < n_h_layers; ++i) {
		if (model == 0) t = kann_layer_rnn(t, n_h_neurons, flag);
		else if (model == 1) t = kann_layer_lstm(t, n_h_neurons, flag);
		else if (model == 2) t = kann_layer_gru(t, n_h_neurons, flag);
		t = kann_layer_dropout(t, h_dropout);
	}
	t = kann_layer_linear(t, n_char);
	t1 = kann_leaf0(KAD_CONST, 1.0f), t1->ext_label = -1; // -1 is for backward compatibility
	t = kad_mul(t, t1); // t1 is the inverse of temperature
	t = kad_softmax(t), t->ext_flag |= KANN_F_OUT;
	t1 = kad_feed(2, 1, n_char), t1->ext_flag |= KANN_F_TRUTH;
	t = kad_ce_multi(t, t1), t->ext_flag |= KANN_F_COST;
	return kann_new(t, 0);
}

int main(int argc, char *argv[])
{
	int c, seed = 11, ulen = 70, n_h_layers = 1, n_h_neurons = 128, model = 2, max_epoch = 50, mbs = 64, c2i[256];
	int len_gen = 1000, use_norm = 1, batch_len = 0, use_batch = 0, use_para = 0;
	float h_dropout = 0.0f, temp = 0.5f, lr = 0.01f, grad_clip = 10.0f;
	kann_t *ann = 0;
	char *fn_in = 0, *fn_out = 0, *prefix = 0;

	while ((c = getopt(argc, argv, "n:l:s:r:m:B:o:i:d:bT:M:u:L:g:Nj:p:P")) >= 0) {
		if (c == 'n') n_h_neurons = atoi(optarg);
		else if (c == 'j') batch_len = atoi(optarg);
		else if (c == 'l') n_h_layers = atoi(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 'i') fn_in = optarg;
		else if (c == 'o') fn_out = optarg;
		else if (c == 'r') lr = atof(optarg);
		else if (c == 'm') max_epoch = atoi(optarg);
		else if (c == 'B') mbs = atoi(optarg);
		else if (c == 'd') h_dropout = atof(optarg);
		else if (c == 'T') temp = atof(optarg);
		else if (c == 'u') ulen = atoi(optarg);
		else if (c == 'L') len_gen = atoi(optarg);
		else if (c == 'g') grad_clip = atof(optarg);
		else if (c == 'N') use_norm = 0;
		else if (c == 'p') prefix = optarg;
		else if (c == 'b') use_batch = 1;
		else if (c == 'P') use_para = use_batch = 1;
		else if (c == 'M') {
			if (strcmp(optarg, "rnn") == 0) model = 0;
			else if (strcmp(optarg, "lstm") == 0) model = 1;
			else if (strcmp(optarg, "gru") == 0) model = 2;
		}
	}
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
		fprintf(fp, "    -g FLOAT    gradient clipping threshold [%g]\n", grad_clip);
		fprintf(fp, "    -j INT      size of a batch [input text length]\n");
		fprintf(fp, "    -b          use minibatch (run faster but converge slower)\n");
		fprintf(fp, "    -P          independent paragraphs (force -b)\n");
		fprintf(fp, "  Text generation:\n");
		fprintf(fp, "    -p STR      prefix []");
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
		tg_train(ann, tg, lr, ulen, mbs, max_epoch, grad_clip, fn_out, batch_len, use_batch, use_para);
		free(tg->data); free(tg);
	} else tg_gen(stdout, ann, temp, len_gen, c2i, prefix);

	kann_delete(ann);
	return 0;
}
