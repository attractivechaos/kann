#include <unistd.h>
#include <stdlib.h>
#include "kann.h"

void kad_print_dot(FILE *fp, int n, kad_node_t **v)
{
	int i, j;
	for (i = 0; i < n; ++i) v[i]->tmp = i;
	fprintf(fp, "digraph {\n");
	for (i = n - 1; i >= 0; --i) {
		kad_node_t *p = v[i];
		if (p->op > 0) fprintf(fp, "\t%d [label=\"%s\"]\n", i, kad_op_name[p->op]);
		for (j = 0; j < p->n_child; ++j)
			fprintf(fp, "\t%d -> %d\n", p->child[j]->tmp, i);
		if (p->pre) fprintf(fp, "\t%d -> %d [style=dotted,weight=0,constraint=false]\n", i, p->pre->tmp);
	}
	fprintf(fp, "}\n");
	for (i = 0; i < n; ++i) v[i]->tmp = 0;
}

int main(int argc, char *argv[])
{
	int c, *len = 0, n_len = 0;
	void (*out_func)(FILE *fp, int n, kad_node_t **v) = kad_print_graph;
	kann_t *ann;

	while ((c = getopt(argc, argv, "dl:")) >= 0)
		if (c == 'l') ++n_len;
	if (n_len) len = (int*)calloc(n_len, sizeof(int));
	optind = 1, n_len = 0;
	while ((c = getopt(argc, argv, "dl:")) >= 0) {
		if (c == 'l') len[n_len++] = atoi(optarg);
		else if (c == 'd') out_func = kad_print_dot;
	}
	if (argc - optind == 0) {
		fprintf(stderr, "Usage: inspect [-l len] <in.kan>\n");
		return 1;
	}
	ann = kann_load(argv[optind]);
	if (len) {
		kann_t *un;
		un = kann_unroll_array(ann, len);
		out_func(stdout, un->n, un->v);
		kann_delete_unrolled(un);
	} else out_func(stdout, ann->n, ann->v);
	kann_delete(ann);
	free(len);
	return 0;
}
