CC=			gcc
CFLAGS=		-g -Wall -Wc++-compat -O2
CPPFLAGS=
INCLUDES=	-I.
EXE=		examples/mlp examples/textgen examples/rnn-bit examples/rnn-bit2 examples/mnist-cnn \
			examples/rnn-lyap examples/inspect
LIBS=		-lm -lz

ifdef CBLAS
	CPPFLAGS+=-DHAVE_CBLAS
	INCLUDES+=-I$(CBLAS)/include
	LIBS+=-L$(CBLAS)/lib -lopenblas
endif

.SUFFIXES:.c .o
.PHONY:all clean depend

.c.o:
		$(CC) -c $(CFLAGS) $(INCLUDES) $(CPPFLAGS) $< -o $@

all:kautodiff.o kann.o kann_extra/kann_data.o $(EXE)

kann_extra/kann_data.o:kann_extra/kann_data.c
		$(CC) -c $(CFLAGS) -DHAVE_ZLIB $< -o $@

examples/mlp:examples/mlp.o kautodiff.o kann.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/textgen:examples/textgen.o kautodiff.o kann.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/rnn-bit:examples/rnn-bit.o kautodiff.o kann.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/rnn-bit2:examples/rnn-bit2.o kautodiff.o kann.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/rnn-lyap:examples/rnn-lyap.o kautodiff.o kann.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/inspect:examples/inspect.o kautodiff.o kann.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

examples/mnist-cnn:examples/mnist-cnn.o kautodiff.o kann.o kann_extra/kann_data.o
		$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
		rm -fr *.o */*.o a.out */a.out *.a *.dSYM */*.dSYM $(EXE)

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c kann_extra/*.c examples/*.c)

# DO NOT DELETE

kann.o: kann.h kautodiff.h
kautodiff.o: kautodiff.h
kann_extra/kann_data.o: kann_extra/kseq.h kann_extra/kann_data.h
examples/inspect.o: kann.h kautodiff.h
examples/mlp.o: kann.h kautodiff.h kann_extra/kann_data.h
examples/mnist-cnn.o: kann_extra/kann_data.h kann.h kautodiff.h
examples/rnn-bit.o: kann.h kautodiff.h
examples/rnn-bit2.o: kann.h kautodiff.h
examples/rnn-lyap.o: kann.h kautodiff.h
examples/textgen.o: kann.h kautodiff.h kann_extra/kseq.h
