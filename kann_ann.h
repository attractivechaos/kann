#ifndef KANN_ANN_H
#define KANN_ANN_H

#define KAD_LABEL_INPUT  1
#define KAD_LABEL_OUTPUT 2

#include "kautodiff.h"

#ifdef __cplusplus
extern "C" {
#endif

void kann_set_batch(int B, int n_node, kad_node_t **node);

#ifdef __cplusplus
}
#endif

#endif
