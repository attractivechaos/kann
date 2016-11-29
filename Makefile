CC=			gcc
CFLAGS=		-g -Wall -Wc++-compat -O2
CPPFLAGS=
EXE=		models/mlp models/textgen examples/rnn-bit
LIBS=		-lm -lz

.SUFFIXES:.c .o
.PHONY:all clean depend

.c.o:
		$(CC) -c $(CFLAGS) -I. $(CPPFLAGS) $< -o $@

all:kautodiff.o kann.o $(EXE)

models/kann_data.o:models/kann_data.c
		$(CC) -c $(CFLAGS) -DHAVE_ZLIB $< -o $@

models/mlp:models/mlp.o kautodiff.o kann.o models/kann_data.o
		$(CC) $(CFLAGS) -o $@ -I. $^ $(LIBS)

models/textgen:models/textgen.o kautodiff.o kann.o
		$(CC) $(CFLAGS) -o $@ -I. $^ $(LIBS)

examples/rnn-bit:examples/rnn-bit.o kautodiff.o kann.o
		$(CC) $(CFLAGS) -o $@ -I. $^ $(LIBS)

clean:
		rm -fr *.o */*.o a.out */a.out *.a *.dSYM */*.dSYM $(EXE)

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c examples/*.c models/*.c)

# DO NOT DELETE

kann.o: kann.h kautodiff.h
kautodiff.o: kautodiff.h
examples/rnn-bit.o: kann.h kautodiff.h
models/kann_data.o: models/kseq.h models/kann_data.h
models/mlp.o: kann.h kautodiff.h models/kann_data.h
models/textgen.o: kann.h kautodiff.h models/kseq.h
