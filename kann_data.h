#ifndef KANN_DATA_H
#define KANN_DATA_H

typedef struct kann_data_t {
	int n_row, n_col, n_grp;
	float **x;
	char **rname, **cname;
	int *grp;
} kann_data_t;

#ifdef __cplusplus
extern "C" {
#endif

kann_data_t *kann_data_read(const char *fn);
void kann_data_free(kann_data_t *d);

#ifdef __cplusplus
}
#endif

#endif
