#ifndef KANN_H
#define KANN_H

#define KANN_VERSION "r117"

#define KANN_L_IN     1
#define KANN_L_OUT    2
#define KANN_L_TRUTH  3
#define KANN_L_COST   4
#define KANN_L_LAST   5

#define KANN_RA_RESET           1
#define KANN_RA_READ_TRAIN      2
#define KANN_RA_READ_VALIDATE   3

#include <stdint.h>
#include "kautodiff.h"

typedef struct {
	float lr;    // learning rate
	float fv;    // fraction of validation data
	int max_mbs; // max mini-batch size
	int max_rnn_len;
	int epoch_lazy;
	int max_epoch;

	float decay;
} kann_mopt_t;

typedef struct {
	int n;
	kad_node_t **v;
	float *t, *g;
} kann_t;

typedef kad_node_t (*kann_activate_f)(kad_node_t*);
typedef int (*kann_reader_f)(void *data, int action, int *len, int max_bs, float **x, float **y);

#define kann_n_par(a) (kad_n_var((a)->n, (a)->v))

extern int kann_verbose;

#ifdef __cplusplus
extern "C" {
#endif

kann_t *kann_init(void);
void kann_delete(kann_t *a);
int kann_n_in(const kann_t *a);
int kann_n_out(const kann_t *a);

void kann_collate_var(kann_t *a);
kann_t *kann_rnn_unroll(kann_t *a, int len, int pool_hidden);

void kann_write(const char *fn, const kann_t *ann);
kann_t *kann_read(const char *fn);

void kann_mopt_init(kann_mopt_t *mo);
void kann_train(const kann_mopt_t *mo, kann_t *a, kann_reader_f rdr, void *data);
void kann_fnn_train(const kann_mopt_t *mo, kann_t *a, int n, float **x, float **y);
const float *kann_fnn_apply1(kann_t *a, float *x);
float *kann_rnn_apply_seq1(kann_t *a, int len, float **x);

kann_t *kann_fnn_gen_mlp(int n_in, int n_out, int n_hidden_layers, int n_hidden_neurons);

kann_t *kann_rnn_gen_vanilla(int n_in, int n_out, int n_hidden_layers, int n_hidden_neurons);
kann_t *kann_rnn_gen_gru(int n_in, int n_out, int n_hidden_layers, int n_hidden_neurons);

void *kann_rdr_xy_new(int n, float frac_validate, int d_x, float **x, int d_y, float **y);
void kann_rdr_xy_delete(void *data);
int kann_rdr_xy_read(void *data, int action, int *len, int max_bs, float **x, float **y);

#ifdef __cplusplus
}
#endif

#endif
