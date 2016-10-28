#ifndef KANN_MIN_H
#define KANN_MIN_H

typedef void (*kann_gradient_f)(int n, const float *x, float *gradient, void *data);

#ifdef __cplusplus
extern "C" {
#endif

void kann_RMSprop(int n, float h0, const float *h, float decay, float *t, float *g, float *r, kann_gradient_f func, void *data);

#ifdef __cplusplus
}
#endif

#endif
