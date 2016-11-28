#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <stdio.h>
#include <zlib.h>
#include "kann.h"
#include "kann_rand.h"
#include "kseq.h"
KSTREAM_INIT(gzFile, gzread, 16384)

#define MAX_CHAR 256

typedef struct {
	int n, m, n_char, no_space, cnt[2], proc[2];
	float frac_val;
	int *len;
	char **s;
	int map[MAX_CHAR];
} textgen_t;

/*****************
 * Reading lines *
 *****************/

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

static inline int kputc(int c, kstring_t *s)
{
	if (s->l + 1 >= s->m) {
		char *tmp;
		s->m = s->l + 2;
		kroundup32(s->m);
		if ((tmp = (char*)realloc(s->s, s->m)))
			s->s = tmp;
		else
			return EOF;
	}
	s->s[s->l++] = c;
	s->s[s->l] = 0;
	return c;
}

static void gen_map(textgen_t *nn)
{
	int i, j;
	memset(nn->map, 0, MAX_CHAR * sizeof(int));
	for (i = 0; i < nn->n; ++i) {
		for (j = 0; j < nn->len[i]; ++j) {
			int c = (unsigned char)nn->s[i][j];
			if (isspace(c) || c == 0)
				c = nn->s[i][j] = ' ';
			++nn->map[c];
		}
	}
	for (i = j = 0; i < MAX_CHAR; ++i)
		nn->map[i] = nn->map[i] > 0? j++ : -1;
	nn->n_char = j;
}

static void push_paragraph(textgen_t *nn, int len, char *s)
{
	if (s == 0) return;
	if (nn->n == nn->m) {
		nn->m = nn->m? nn->m<<1 : 16;
		nn->len = (int*)realloc(nn->len, nn->m * sizeof(int));
		nn->s = (char**)realloc(nn->s, nn->m * sizeof(char*));
	}
	nn->s[nn->n] = strdup(s);
	nn->len[nn->n++] = len;
}

static textgen_t *read_file(const char *fn, int no_space)
{
	gzFile fp;
	textgen_t *nn;
	kstream_t *ks;
	int dret, i;
	kstring_t s = {0,0,0}, p = {0,0,0};

	fp = fn && strcmp(fn, "-")? gzopen(fn, "r") : gzdopen(fileno(stdin), "r");
	ks = ks_init(fp);
	nn = (textgen_t*)calloc(1, sizeof(textgen_t));
	nn->no_space = no_space;
	nn->frac_val = 0.1f;
	nn->cnt[0] = 10000, nn->cnt[1] = 1000;
	while (ks_getuntil(ks, KS_SEP_LINE, &s, &dret) >= 0) {
		if (s.l > 0 && s.s[s.l-1] == '\r') // for \r\n
			--s.l, s.s[s.l-1] = 0;
		if (s.l) {
			if (!no_space) kputc(' ', &p);
			kputsn(s.s, s.l, &p);
		} else {
			push_paragraph(nn, p.l, p.s);
			p.l = 0;
		}
	}
	push_paragraph(nn, p.l, p.s);
	free(p.s); free(s.s);
	ks_destroy(ks);
	gzclose(fp);
	gen_map(nn);

	for (i = nn->n; i > 1; --i) { // shuffle
		char *tmp;
		int j, len;
		j = (int)(kann_drand() * i);
		tmp = nn->s[j], nn->s[j] = nn->s[i-1], nn->s[i-1] = tmp;
		len = nn->len[j], nn->len[j] = nn->len[i-1], nn->len[i-1] = len;
	}
	fprintf(stderr, "%d characters; %d paragraphs\n", nn->n_char, nn->n);
	return nn;
}

static void textgen_delete(textgen_t *tg)
{
	int i;
	for (i = 0; i < tg->n; ++i) free(tg->s[i]);
	free(tg->s); free(tg->len);
	free(tg);
}

/***************
 * Data reader *
 ***************/

static int textgen_reader(void *data, int action, int len, float *x1, float *y1)
{
	textgen_t *nn = (textgen_t*)data;
	int n[2];
	n[1] = (int)(nn->n * nn->frac_val + .499);
	n[0] = nn->n - n[1];
	if (action == KANN_RDR_BATCH_RESET) {
		nn->proc[0] = nn->proc[1] = 0;
	} else if (action == KANN_RDR_READ_TRAIN || action == KANN_RDR_READ_VALIDATE) {
		int i, j, a, k = action == KANN_RDR_READ_TRAIN? 0 : 1;
		if (nn->proc[k] == nn->cnt[k]) return 0;
		for (;;) {
			j = (int)(kann_drand() * n[k]) + (k? n[0] : 0);
			if (nn->len[j] < len + 1) continue;
			i = (int)((nn->len[j] - len - 1) * kann_drand());
			if (!nn->no_space) {
				for (; i < nn->len[j] && nn->s[j][i] == ' '; ++i);
				for (; i >= 0 && nn->s[j][i] != ' '; --i);
				++i;
			}
			if (i + len < nn->len[j]) break;
		}
		memset(x1, 0, len * nn->n_char * sizeof(float));
		memset(y1, 0, len * nn->n_char * sizeof(float));
		for (a = 0; a < len; ++a) {
			x1[a*nn->n_char + nn->map[(unsigned char)nn->s[j][i+a]]] = 1.0f;
			y1[a*nn->n_char + nn->map[(unsigned char)nn->s[j][i+a+1]]] = 1.0f;
		}
		++nn->proc[k];
		return len;
	}
	return 0;
}

static void textgen_write(const char *fn, kann_t *ann, int map[MAX_CHAR])
{
	FILE *fp;
	fp = fn && strcmp(fn, "-")? fopen(fn, "wb") : stdout;
	kann_write_core(fp, ann);
	fwrite(map, sizeof(int), MAX_CHAR, fp);
	fclose(fp);
}

static kann_t *textgen_read(const char *fn, int map[MAX_CHAR])
{
	FILE *fp;
	kann_t *ann;
	fp = fn && strcmp(fn, "-")? fopen(fn, "rb") : stdin;
	ann = kann_read_core(fp);
	fread(map, sizeof(int), MAX_CHAR, fp);
	fclose(fp);
	return ann;
}

static kann_t *model_gen(int use_gru, int n_char, int n_h_layers, int n_h_neurons, float h_dropout)
{
	int i;
	kad_node_t *t;
	t = kann_layer_input(n_char);
	for (i = 0; i < n_h_layers; ++i) {
		t = use_gru? kann_layer_gru(t, n_h_neurons) : kann_layer_rnn(t, n_h_neurons, kad_tanh);
		t = kann_layer_dropout(t, h_dropout);
	}
	return kann_layer_final(t, n_char, KANN_C_CE);
}

int main(int argc, char *argv[])
{
	int i, c, seed = 11, no_space = 0, n_h_layers = 1, n_h_neurons = 100, use_gru = 0, batch_size = 11000, map[MAX_CHAR];
	float h_dropout = 0.1f, temp = 0.5f;
	kann_t *ann = 0;
	kann_mopt_t mo;
	char *fn_in = 0, *fn_out = 0;

	kann_mopt_init(&mo);
	mo.max_rnn_len = 100;
	mo.lr = 0.01f;
	while ((c = getopt(argc, argv, "n:l:s:r:m:B:o:i:d:gt:Sb:T:")) >= 0) {
		if (c == 'n') n_h_neurons = atoi(optarg);
		else if (c == 'l') n_h_layers = atoi(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 'i') fn_in = optarg;
		else if (c == 'o') fn_out = optarg;
		else if (c == 'r') mo.lr = atof(optarg);
		else if (c == 'm') mo.max_epoch = atoi(optarg);
		else if (c == 'B') mo.max_mbs = atoi(optarg);
		else if (c == 'd') h_dropout = atof(optarg);
		else if (c == 'g') use_gru = 1;
		else if (c == 't') mo.max_rnn_len = atoi(optarg);
		else if (c == 'S') no_space = 1;
		else if (c == 'b') batch_size = atoi(optarg);
		else if (c == 'T') temp = atof(optarg);
	}
	if (argc == optind && fn_in == 0) {
		FILE *fp = stdout;
		fprintf(fp, "Usage: textgen [options] [in.txt]\n");
		fprintf(fp, "Options:\n");
		fprintf(fp, "  Model construction:\n");
		fprintf(fp, "    -i FILE     read trained model from FILE []\n");
		fprintf(fp, "    -o FILE     save trained model to FILE []\n");
		fprintf(fp, "    -s INT      random seed [%d]\n", seed);
		fprintf(fp, "    -l INT      number of hidden layers [%d]\n", n_h_layers);
		fprintf(fp, "    -n INT      number of hidden neurons per layer [%d]\n", n_h_neurons);
		fprintf(fp, "    -g          use GRU (vanilla RNN by default)\n");
		fprintf(fp, "  Model training:\n");
		fprintf(fp, "    -r FLOAT    learning rate [%g]\n", mo.lr);
		fprintf(fp, "    -d FLOAT    dropout at the hidden layer(s) [%g]\n", h_dropout);
		fprintf(fp, "    -m INT      max number of epochs [%d]\n", mo.max_epoch);
		fprintf(fp, "    -B INT      mini-batch size [%d]\n", mo.max_mbs);
		fprintf(fp, "    -t INT      max unroll [%d]\n", mo.max_rnn_len);
		fprintf(fp, "  Text generation:\n");
		fprintf(fp, "    -T FLOAT    temperature [%g]\n", temp);
		return 1;
	}

	kann_srand(seed);
	kad_trap_fe();
	if (fn_in) ann = textgen_read(fn_in, map);

	if (argc - optind >= 1) { // train
		textgen_t *tg;
		tg = read_file(argv[optind], no_space);
		tg->cnt[0] = batch_size * 1.0f / (1.0f + tg->frac_val);
		tg->cnt[1] = batch_size - tg->cnt[0];
		memcpy(map, tg->map, MAX_CHAR * sizeof(int));
		if (ann) assert(kann_n_in(ann) == tg->n_char && kann_n_out(ann) == tg->n_char);
		else ann = model_gen(use_gru, tg->n_char, n_h_layers, n_h_neurons, h_dropout);
		kann_train(&mo, ann, textgen_reader, tg);
		if (fn_out) textgen_write(fn_out, ann, tg->map);
		textgen_delete(tg);
	} else { // apply
		int n_char, revmap[MAX_CHAR];
		memset(revmap, 0, MAX_CHAR * sizeof(int));
		for (i = 0; i < MAX_CHAR; ++i)
			if (map[i] >= 0) revmap[map[i]] = i;
		n_char = kann_n_in(ann);
		kann_set_hyper(ann, KANN_H_TEMP, temp);
		kann_rnn_start(ann);
		c = revmap[(int)(n_char * kann_drand())];
		for (i = 0; i < 1000; ++i) {
			float x[MAX_CHAR], s, r;
			const float *y;
			int j;
			memset(x, 0, n_char * sizeof(float));
			x[map[c]] = 1.0f;
			y = kann_apply1(ann, x);
			r = kann_drand();
			for (j = 0, s = 0.0f; j < n_char; ++j)
				if (s + y[j] >= r) break;
				else s += y[j];
			c = revmap[j];
			putchar(c);
		}
		putchar('\n');
		kann_rnn_end(ann);
	}

	kann_delete(ann);
	return 0;
}
