#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
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
} charnn_t;

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

static void gen_map(charnn_t *nn)
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

static void push_paragraph(charnn_t *nn, int len, char *s)
{
	if (nn->n == nn->m) {
		nn->m = nn->m? nn->m<<1 : 16;
		nn->len = (int*)realloc(nn->len, nn->m * sizeof(int));
		nn->s = (char**)realloc(nn->s, nn->m * sizeof(char*));
	}
	nn->s[nn->n] = strdup(s);
	nn->len[nn->n++] = len;
}

static charnn_t *read_file(const char *fn, int no_space)
{
	gzFile fp;
	charnn_t *nn;
	kstream_t *ks;
	int dret, i;
	kstring_t s = {0,0,0}, p = {0,0,0};

	fp = fn && strcmp(fn, "-")? gzopen(fn, "r") : gzdopen(fileno(stdin), "r");
	ks = ks_init(fp);
	nn = (charnn_t*)calloc(1, sizeof(charnn_t));
	nn->no_space = no_space;
	nn->frac_val = 0.1f;
	nn->cnt[0] = 10000, nn->cnt[1] = 1000;
	while (ks_getuntil(ks, KS_SEP_LINE, &s, &dret) >= 0) {
		if (s.s[s.l-1] == '\r') // for \r\n
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

static int charnn_reader(void *data, int action, int len, float *x1, float *y1)
{
	charnn_t *nn = (charnn_t*)data;
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

int main(int argc, char *argv[])
{
	int c, no_space = 0, n_h_layers = 1, n_h_neurons = 100, max_unroll = 100, use_vanilla = 0;
	uint64_t seed = 11;
	charnn_t *nn;
	kann_t *ann;
	kann_mopt_t mo;
	char *fn_in = 0, *fn_out = 0;

	kann_mopt_init(&mo);
	mo.max_epoch = 50;
	while ((c = getopt(argc, argv, "i:o:vl:m:n:r:")) >= 0) {
		if (c == 'i') fn_in = optarg;
		else if (c == 'o') fn_out = optarg;
		else if (c == 'v') use_vanilla = 1;
		else if (c == 'l') max_unroll = atoi(optarg);
		else if (c == 'm') mo.max_epoch = atoi(optarg);
		else if (c == 'h') n_h_layers = atoi(optarg);
		else if (c == 'n') n_h_neurons = atoi(optarg);
		else if (c == 'r') mo.lr = atof(optarg);
	}
	if (argc == optind) {
		FILE *fp = stdout;
		fprintf(fp, "Usage: charnn [options] <in.txt>\nOptions:\n");
		fprintf(fp, "  -i FILE     read the model from FILE []\n");
		fprintf(fp, "  -o FILE     write the model to FILE []\n");
		fprintf(fp, "  -v          use vanilla RNN instead of GRU\n");
		fprintf(fp, "  -l INT      max RNN unroll length [%d]\n", max_unroll);
		fprintf(fp, "  -h INT      number of hidden layers [%d]\n", n_h_layers);
		fprintf(fp, "  -n INT      number of neurons per hidden layer [%d]\n", n_h_neurons);
		return 1;
	}

	kann_srand(seed);
	nn = read_file(argv[optind], no_space);
	if (fn_in) {
		ann = kann_read(fn_in);
	} else {
		int i;
		kad_node_t *t;
		t = kann_layer_input(nn->n_char);
		for (i = 0; i < n_h_layers; ++i)
			t = use_vanilla? kann_layer_rnn(t, n_h_neurons) : kann_layer_gru(t, n_h_neurons);
		ann = kann_layer_final(t, nn->n_char, KANN_C_CE);
	}
	mo.max_rnn_len = max_unroll;
	kann_train(&mo, ann, charnn_reader, nn);
	if (fn_out) kann_write(fn_out, ann);
	kann_delete(ann);
	return 0;
}
