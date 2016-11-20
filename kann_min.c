#include <math.h>
#include "kann_min.h"

#ifdef __SSE__
#include <xmmintrin.h>

void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r)
{
	int i, n4 = n>>2<<2;
	__m128 vh, vg, vr, vt, vd, vd1, tmp, vtiny;
	vh = _mm_set1_ps(h0);
	vd = _mm_set1_ps(decay);
	vd1 = _mm_set1_ps(1.0f - decay);
	vtiny = _mm_set1_ps(1e-6f);
	for (i = 0; i < n4; i += 4) {
		vt = _mm_loadu_ps(&t[i]);
		vr = _mm_loadu_ps(&r[i]);
		vg = _mm_loadu_ps(&g[i]);
		if (h) vh = _mm_loadu_ps(&h[i]);
		vr = _mm_add_ps(_mm_mul_ps(vd1, _mm_mul_ps(vg, vg)), _mm_mul_ps(vd, vr));
		_mm_storeu_ps(&r[i], vr);
		tmp = _mm_sub_ps(vt, _mm_mul_ps(_mm_mul_ps(vh, _mm_rsqrt_ps(_mm_add_ps(vtiny, vr))), vg));
		_mm_storeu_ps(&t[i], tmp);
	}
	for (; i < n; ++i) {
		r[i] = (1. - decay) * g[i] * g[i] + decay * r[i];
		t[i] -= (h? h[i] : h0) / sqrt(1e-6 + r[i]) * g[i];
	}
}
#else
void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r)
{
	int i;
	for (i = 0; i < n; ++i) {
		float lr = h? h[i] : h0;
		r[i] = (1. - decay) * g[i] * g[i] + decay * r[i];
		t[i] -= lr / sqrt(1e-6 + r[i]) * g[i];
	}
}
#endif

kann_min_t *kann_min_new(int mini_algo, int batch_algo, int n)
{
	kann_min_t *m;
	if (mini_algo <= 0) mini_algo = KANN_MM_RMSPROP;
	if (batch_algo <= 0) batch_algo = KANN_MB_CONST;
	m = (kann_min_t*)calloc(1, sizeof(kann_min_t));
	m->mini_algo = mini_algo, m->batch_algo = batch_algo, m->n = n;
	if (mini_algo == KANN_MM_RMSPROP) {
		m->lr = 0.001f, m->decay = 0.9f;
		m->maux = (float*)calloc(n, sizeof(float));
	}
	return m;
}

void kann_min_delete(kann_min_t *m)
{
	free(m->maux); free(m->baux); free(m);
}

void kann_min_mini_update(kann_min_t *m, const float *g, float *t)
{
	if (m->mini_algo == KANN_MM_RMSPROP)
		kann_RMSprop(m->n, m->lr, 0, m->decay, g, t, m->maux);
}

void kann_min_batch_finish(kann_min_t *m, const float *t)
{
}
