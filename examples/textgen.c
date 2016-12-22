#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include "kann.h"

typedef struct {
	int len, n_char;
	uint8_t *data;
	int tot[2], n_proc[2];
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

tg_data_t *tg_read(const char *fn)
{
	int i, j;
	tg_data_t *tg;
	tg = (tg_data_t*)calloc(1, sizeof(tg_data_t));
	tg->data = tg_read_file(fn, &tg->len);
	for (i = 0; i < tg->len; ++i)
		tg->c2i[tg->data[i]] = 1;
	for (i = j = 0; i < 256; ++i)
		if (tg->c2i[i] == 0) tg->c2i[i] = -1;
		else tg->c2i[i] = j++;
	tg->n_char = j;
	for (i = 0; i < tg->len; ++i)
		tg->data[i] = tg->c2i[tg->data[i]];
	return tg;
}

void tg_save(const char *fn, kann_t *ann, int map[256])
{
	FILE *fp;
	fp = fn && strcmp(fn, "-")? fopen(fn, "wb") : stdout;
	kann_save_fp(fp, ann);
	fwrite(map, sizeof(int), 256, fp);
	fclose(fp);
}

kann_t *tg_load(const char *fn, int map[256])
{
	FILE *fp;
	kann_t *ann;
	fp = fn && strcmp(fn, "-")? fopen(fn, "rb") : stdin;
	ann = kann_load_fp(fp);
	fread(map, sizeof(int), 256, fp);
	fclose(fp);
	return ann;
}

int tg_reader(void *data, int action, int len, float *x1, float *y1)
{
	tg_data_t *tg = (tg_data_t*)data;
	if (action == KANN_RDR_BATCH_RESET) {
		tg->n_proc[0] = tg->n_proc[1] = 0;
	} else if (action == KANN_RDR_READ_TRAIN || action == KANN_RDR_READ_VALIDATE) {
		int i, a, k = action == KANN_RDR_READ_TRAIN? 0 : 1;
		if (tg->n_proc[k] == tg->tot[k]) return 0;
		i = (int)(kann_drand() * (tg->len - len)) + 1;
		memset(x1, 0, len * tg->n_char * sizeof(float));
		memset(y1, 0, len * tg->n_char * sizeof(float));
		for (a = 0; a < len; ++a) {
			x1[a * tg->n_char + tg->data[a+i-1]] = 1.0f;
			y1[a * tg->n_char + tg->data[a+i]] = 1.0f;
		}
		++tg->n_proc[k];
		return len;
	}
	return 0;
}

static kann_t *model_gen(int model, int n_char, int n_h_layers, int n_h_neurons, float h_dropout)
{
	int i;
	kad_node_t *t;
	t = kann_layer_input(n_char);
	for (i = 0; i < n_h_layers; ++i) {
		if (model == 0) t = kann_layer_rnn(t, n_h_neurons, kad_tanh);
		else if (model == 1) t = kann_layer_lstm(t, n_h_neurons);
		else if (model == 2) t = kann_layer_gru(t, n_h_neurons);
		t = kann_layer_dropout(t, h_dropout);
	}
	return kann_layer_final(t, n_char, KANN_C_CEM);
}

int main(int argc, char *argv[])
{
	int i, c, seed = 11, n_h_layers = 1, n_h_neurons = 128, model = 0, batch_size = 1000000, c2i[256];
	float h_dropout = 0.0f, temp = 0.5f;
	kann_t *ann = 0;
	kann_mopt_t mo;
	char *fn_in = 0, *fn_out = 0;

	kann_mopt_init(&mo);
	mo.max_rnn_len = 100;
	mo.lr = 0.01f;
	while ((c = getopt(argc, argv, "n:l:s:r:m:B:o:i:d:t:b:T:M:R")) >= 0) {
		if (c == 'n') n_h_neurons = atoi(optarg);
		else if (c == 'l') n_h_layers = atoi(optarg);
		else if (c == 's') seed = atoi(optarg);
		else if (c == 'i') fn_in = optarg;
		else if (c == 'o') fn_out = optarg;
		else if (c == 'r') mo.lr = atof(optarg);
		else if (c == 'm') mo.max_epoch = atoi(optarg);
		else if (c == 'B') mo.max_mbs = atoi(optarg);
		else if (c == 'd') h_dropout = atof(optarg);
		else if (c == 't') mo.max_rnn_len = atoi(optarg);
		else if (c == 'b') batch_size = atoi(optarg);
		else if (c == 'T') temp = atof(optarg);
		else if (c == 'R') mo.batch_algo = KANN_MB_iRprop;
		else if (c == 'M') {
			if (strcmp(optarg, "rnn") == 0) model = 0;
			else if (strcmp(optarg, "lstm") == 0) model = 1;
			else if (strcmp(optarg, "gru") == 0) model = 2;
		}
	}
	if (argc == optind && fn_in == 0) {
		FILE *fp = stdout;
		fprintf(fp, "Usage: textgen [options] <in.fq>\n");
		fprintf(fp, "Options:\n");
		fprintf(fp, "  Model construction:\n");
		fprintf(fp, "    -i FILE     read trained model from FILE []\n");
		fprintf(fp, "    -o FILE     save trained model to FILE []\n");
		fprintf(fp, "    -s INT      random seed [%d]\n", seed);
		fprintf(fp, "    -l INT      number of hidden layers [%d]\n", n_h_layers);
		fprintf(fp, "    -n INT      number of hidden neurons per layer [%d]\n", n_h_neurons);
		fprintf(fp, "    -M STR      model: rnn, lstm or gru [rnn]\n");
		fprintf(fp, "  Model training:\n");
		fprintf(fp, "    -r FLOAT    learning rate [%g]\n", mo.lr);
		fprintf(fp, "    -d FLOAT    dropout at the hidden layer(s) [%g]\n", h_dropout);
		fprintf(fp, "    -m INT      max number of epochs [%d]\n", mo.max_epoch);
		fprintf(fp, "    -B INT      mini-batch size [%d]\n", mo.max_mbs);
		fprintf(fp, "    -t INT      max unroll [%d]\n", mo.max_rnn_len);
		fprintf(fp, "    -R          use iRprop- after a batch\n");
		fprintf(fp, "  Text generation:\n");
		fprintf(fp, "    -T FLOAT    temperature [%g]\n", temp);
		return 1;
	}

	kann_srand(seed);
	kad_trap_fe();
	if (fn_in) ann = tg_load(fn_in, c2i);

	if (argc - optind >= 1) { // train
		tg_data_t *tg;
		tg = tg_read(argv[optind]);
		fprintf(stderr, "Warning: 'validation cost' below is not right as validation not separated from training data.\n");
		fprintf(stderr, "Read %d characters; alphabet size %d\n", tg->len, tg->n_char);
		tg->tot[0] = batch_size / mo.max_rnn_len;
		tg->tot[1] = (int)(batch_size / mo.max_rnn_len * mo.fv);
		if (!ann) ann = model_gen(model, tg->n_char, n_h_layers, n_h_neurons, h_dropout);
		kann_train(&mo, ann, tg_reader, tg);
		if (fn_out) tg_save(fn_out, ann, tg->c2i);
		free(tg->data); free(tg);
	} else { // apply
		int n_char, i2c[256];
		memset(i2c, 0, 256 * sizeof(int));
		for (i = 0; i < 256; ++i)
			if (c2i[i] >= 0) i2c[c2i[i]] = i;
		n_char = kann_n_in(ann);
		kann_set_by_flag(ann, KANN_F_TEMP, 1.0f/temp);
		kann_rnn_start(ann);
		c = (int)(n_char * kann_drand());
		for (i = 0; i < 1000; ++i) {
			float x[256], s, r;
			const float *y;
			memset(x, 0, n_char * sizeof(float));
			x[c] = 1.0f;
			y = kann_apply1(ann, x);
			r = kann_drand();
			for (c = 0, s = 0.0f; c < n_char; ++c)
				if (s + y[c] >= r) break;
				else s += y[c];
			putchar(i2c[c]);
		}
		putchar('\n');
		kann_rnn_end(ann);
	}

	kann_delete(ann);
	return 0;
}
