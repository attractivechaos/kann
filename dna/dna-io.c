#include <zlib.h>
#include <stdio.h>
#include "dna-io.h"
#include "kseq.h"
KSEQ_INIT2(, gzFile, gzread)

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

void dn_seq2vec_ds(int l, const uint8_t *seq4, float *x)
{
	int i, c;
	uint8_t *rev4;
	rev4 = (uint8_t*)malloc(l);
	for (i = 0; i < l; ++i) rev4[l-i-1] = seq4[i] > 3? 4 : 3 - seq4[i];
	memset(x, 0, 8 * l * sizeof(float));
	for (c = 0; c < 4; ++c) {
		float *x1 = &x[c * l];
		for (i = 0; i < l; ++i)
			if (seq4[i] == c) x1[i] = 1.0f;
		x1 = &x[(c + 4) * l];
		for (i = 0; i < l; ++i)
			if (rev4[i] == c) x1[i] = 1.0f;
	}
	free(rev4);
}

dn_seqs_t *dn_read(const char *fn)
{
	gzFile fp;
	kseq_t *ks;
	dn_seqs_t *s;
	int i, max_lbl = 0;

	fp = fn && strcmp(fn, "-")? gzopen(fn, "r") : gzdopen(fileno(stdin), "r");
	ks = kseq_init(fp);
	s = (dn_seqs_t*)calloc(1, sizeof(dn_seqs_t));
	while (kseq_read(ks) >= 0) {
		for (i = 0; i < ks->seq.l; ++i) {
			ks->seq.s[i] = seq_nt4_table[(int)ks->seq.s[i]];
			if (ks->qual.l == ks->seq.l)
				ks->qual.s[i] = ks->qual.s[i] - 33;
		}
		if (s->n == s->m) {
			s->m = s->m? s->m<<1 : 16;
			s->len = (int*)realloc(s->len, s->m * sizeof(int));
			s->seq = (uint8_t**)realloc(s->seq, s->m * sizeof(uint8_t*));
			s->lbl = (uint8_t**)realloc(s->lbl, s->m * sizeof(uint8_t*));
		}
		s->len[s->n] = ks->seq.l;
		s->seq[s->n] = (uint8_t*)malloc(ks->seq.l);
		memcpy(s->seq[s->n], ks->seq.s, ks->seq.l);
		if (ks->qual.l == ks->seq.l) {
			s->lbl[s->n] = (uint8_t*)malloc(ks->qual.l);
			memcpy(s->lbl[s->n], ks->qual.s, ks->qual.l);
			for (i = 0; i < ks->qual.l; ++i)
				max_lbl = max_lbl > ks->qual.s[i]? max_lbl : ks->qual.s[i];
		}
		++s->n;
	}
	s->n_lbl = max_lbl + 1;
	kseq_destroy(ks);
	gzclose(fp);
	return s;
}

void dn_destroy(dn_seqs_t *s)
{
	int i;
	for (i = 0; i < s->n; ++i) {
		free(s->seq[i]);
		if (s->lbl) free(s->lbl[i]);
	}
	free(s->seq); free(s->lbl);
	free(s);
}
