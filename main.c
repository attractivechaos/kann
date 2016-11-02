#include <sys/resource.h>
#include <sys/time.h>
#include <string.h>
#include <stdio.h>
#include "kann.h"

int main_mlp_train(int argc, char *argv[]);

void liftrlimit()
{
#ifdef __linux__
	struct rlimit r;
	getrlimit(RLIMIT_AS, &r);
	r.rlim_cur = r.rlim_max;
	setrlimit(RLIMIT_AS, &r);
#endif
}

double cputime()
{
	struct rusage r;
	getrusage(RUSAGE_SELF, &r);
	return r.ru_utime.tv_sec + r.ru_stime.tv_sec + 1e-6 * (r.ru_utime.tv_usec + r.ru_stime.tv_usec);
}

double realtime()
{
	struct timeval tp;
	struct timezone tzp;
	gettimeofday(&tp, &tzp);
	return tp.tv_sec + tp.tv_usec * 1e-6;
}

#ifdef __SSE__
#include <xmmintrin.h>
#endif

int main(int argc, char *argv[])
{
	int ret = 0, i;
	double t_start;
#ifdef __SSE__
	_MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~(_MM_MASK_INVALID | _MM_MASK_DIV_ZERO));
#endif
	liftrlimit();
	if (argc == 1) {
		fprintf(stderr, "Usage: kann <command> <arguments>\n");
		fprintf(stderr, "Commands:\n");
		fprintf(stderr, "  mlp-train     train a multi-layer perceptron\n");
		fprintf(stderr, "  version       show version number\n");
		return 1;
	}
	t_start = realtime();
	if (strcmp(argv[1], "mlp-train") == 0) ret = main_mlp_train(argc-1, argv+1);
	else if (strcmp(argv[1], "version") == 0) {
		puts(KANN_VERSION);
		return 0;
	} else {
		fprintf(stderr, "[E::%s] unknown command\n", __func__);
		return 1;
	}
	if (ret == 0) {
		fprintf(stderr, "[M::%s] Version: %s\n", __func__, KANN_VERSION);
		fprintf(stderr, "[M::%s] CMD:", __func__);
		for (i = 0; i < argc; ++i)
			fprintf(stderr, " %s", argv[i]);
		fprintf(stderr, "\n[M::%s] Real time: %.3f sec; CPU: %.3f sec\n", __func__, realtime() - t_start, cputime());
	}
	return ret;
}
