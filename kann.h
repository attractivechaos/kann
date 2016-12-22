/*
  The MIT License

  Copyright (c) 2016  Broad Institute

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
  ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#ifndef KANN_H
#define KANN_H

#define KANN_VERSION "r305"

#define KANN_F_IN       0x1   // input
#define KANN_F_OUT      0x2   // output
#define KANN_F_TRUTH    0x4   // truth output
#define KANN_F_COST     0x8   // final cost
#define KANN_F_TEMP     0x10  // temperature for softmax
#define KANN_F_DROPOUT  0x20  // dropout ratio

#define KANN_C_CEB      1   // binary cross-entropy cost, used with sigmoid
#define KANN_C_CEM      2   // multi-class cross-entropy cost, used with softmax

#define KANN_RDR_BATCH_RESET     1
#define KANN_RDR_MINI_RESET      2
#define KANN_RDR_READ_TRAIN      3
#define KANN_RDR_READ_VALIDATE   4

#define KANN_MM_SGD     1
#define KANN_MM_RMSPROP 2

#define KANN_MB_CONST   1
#define KANN_MB_iRprop  2

#include "kautodiff.h"

typedef struct {
	int n;
	kad_node_t **v;
	float *x, *g, *c;
} kann_t;

typedef struct {
	float lr;    // learning rate
	float fv;    // fraction of validation data
	int max_mbs; // max mini-batch size
	int max_rnn_len;
	int epoch_lazy;
	int max_epoch;
	int mini_algo, batch_algo;
} kann_mopt_t;

typedef int (*kann_reader_f)(void *data, int action, int max_len, float *x, float *y);

extern int kann_verbose;

#ifdef __cplusplus
extern "C" {
#endif

// common layers
kad_node_t *kann_layer_input(int n1);
kad_node_t *kann_layer_linear(kad_node_t *in, int n1);
kad_node_t *kann_layer_dropout(kad_node_t *t, float r);
kad_node_t *kann_layer_rnn(kad_node_t *in, int n1, kad_node_t *(*af)(kad_node_t*));
kad_node_t *kann_layer_lstm(kad_node_t *in, int n1);
kad_node_t *kann_layer_gru(kad_node_t *in, int n1);
kad_node_t *kann_layer_conv2d(kad_node_t *in, int n_flt, int k_rows, int k_cols, int stride, int pad);
kann_t *kann_gen(kad_node_t *cost);
kann_t *kann_layer_final(kad_node_t *t, int n_out, int cost_type);

kad_node_t *kann_new_weight(int n_row, int n_col);
kad_node_t *kann_new_bias(int n);
kad_node_t *kann_new_weight_conv2d(int n_out_channel, int n_in_channel, int k_row, int k_col);
kad_node_t *kann_new_weight_conv1d(int n_out, int n_in, int kernel_len);

// basic model operations
void kann_set_batch_size(kann_t *a, int B);
void kann_set_scalar(kann_t *a, int flag, float z);
int kann_bind_feed(kann_t *a, int flag, float **x);
void kann_delete(kann_t *a);
float kann_cost(kann_t *a, int cal_grad);

// number of input and output variables
int kann_n_in(const kann_t *a);
int kann_n_out(const kann_t *a);

// unroll an RNN to an FNN
int kann_is_rnn(const kann_t *a);
kann_t *kann_rnn_unroll(kann_t *a, int len);
void kann_delete_unrolled(kann_t *a);

// train a model
void kann_mopt_init(kann_mopt_t *mo);
void kann_train(const kann_mopt_t *mo, kann_t *a, kann_reader_f rdr, void *data);
void kann_fnn_train(const kann_mopt_t *mo, kann_t *a, int n, float **x, float **y);

// apply a trained model
const float *kann_apply1(kann_t *a, float *x);
void kann_rnn_start(kann_t *a);
void kann_rnn_end(kann_t *a);
float *kann_rnn_apply_seq1(kann_t *a, int len, float *x);
void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r);

// model I/O
void kann_save_fp(FILE *fp, kann_t *ann);
void kann_save(const char *fn, kann_t *ann);
kann_t *kann_load_fp(FILE *fp);
kann_t *kann_load(const char *fn);

// pseudo-random number generator
void kann_srand(uint64_t seed);
uint64_t kann_rand(void);
double kann_drand(void);
double kann_normal(void);
void kann_normal_array(float sigma, int n, float *x);

#ifdef __cplusplus
}
#endif

#endif
