CC=			gcc
CFLAGS=		-g -Wall -pg -mavx -Wextra -Wc++-compat -O2
CFLAGS_LIB=	#-ansi -pedantic -Wno-long-long # ANSI C does not have inline which affects performance a little bit
CPPFLAGS=	-DHAVE_PTHREAD
INCLUDES=	-I.
EXE=		examples/mnist-cnn
EXE_OPT=		examples/mnist-cnn_opt
LIBS=		-lpthread -lz -lm
ifdef CBLAS
	CPPFLAGS+=-DHAVE_CBLAS
	INCLUDES+=-I$(CBLAS)/include
	LIBS=-fopenmp -pthread -L$(CBLAS)/lib -lopenblas -lz -lm
endif

.SUFFIXES:.c .o
.PHONY:all clean depend

.c.o:
		$(CC) -c $(CFLAGS) $(INCLUDES) $(CPPFLAGS) $< -o $@

all:kautodiff.o kann.o kann_extra/kann_data.o $(EXE)

opt:kautodiff_opt.o kann_opt.o kann_extra/kann_data.o $(EXE_OPT)

kautodiff.o:kautodiff.c
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

kautodiff_opt.o:kautodiff_opt.c
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

kann.o:kann.c
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

kann_opt.o:kann_opt.c
		$(CC) -c $(CFLAGS) $(CFLAGS_LIB) $(INCLUDES) $(CPPFLAGS) -o $@ $<

kann_extra/kann_data.o:kann_extra/kann_data.c
		$(CC) -c $(CFLAGS) -DHAVE_ZLIB $< -o $@

examples/mnist-cnn:examples/mnist-cnn.o kautodiff.o kann.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/mnist-cnn_opt:examples/mnist-cnn_opt.o kautodiff_opt.o kann_opt.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
		rm -fr *.o */*.o a.out */a.out *.a *.dSYM */*.dSYM $(EXE)

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c kann_extra/*.c examples/*.c)

# DO NOT DELETE

kann.o: kann.h kautodiff.h
kann_opt.o: kann.h kautodiff.h
kautodiff.o: kautodiff.h
kautodiff_opt.o: kautodiff_opt.h
kann_extra/kann_data.o: kann_extra/kseq.h kann_extra/kann_data.h
examples/mnist-cnn.o: kann_extra/kann_data.h kann.h kautodiff.h
examples/mnist-cnn_opt.o: kann_extra/kann_data.h kann_opt.h kautodiff_opt.h
