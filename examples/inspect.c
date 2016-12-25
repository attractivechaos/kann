#include <unistd.h>
#include <stdlib.h>
#include "kann.h"

int main(int argc, char *argv[])
{
	int c, len = 0;
	kann_t *ann;
	while ((c = getopt(argc, argv, "l:")) >= 0)
		if (c == 'l') len = atoi(optarg);
	if (argc - optind == 0) {
		fprintf(stderr, "Usage: inspect [-l len] <in.knm>\n");
		return 1;
	}
	ann = kann_load(argv[optind]);
	if (len > 0) {
		kann_t *un;
		un = kann_unroll(ann, len);
		kad_print_graph(stdout, un->n, un->v);
		kann_delete_unrolled(un);
	} else kad_print_graph(stdout, ann->n, ann->v);
	kann_delete(ann);
	return 0;
}
