CC=			gcc
CFLAGS=		-g -Wall -Wc++-compat -O2
CPPFLAGS=
ZLIB_FLAGS=	-DHAVE_ZLIB   # comment out this line to drop the zlib dependency
INCLUDES=	-I.
EXE=		models/mlp models/textgen examples/rnn-bit
LIBS=		-lm -lz

.SUFFIXES:.c .o
.PHONY:all demo clean depend

.c.o:
		$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

all:$(EXE)

kann_data.o:kann_data.c
		$(CC) -c $(CFLAGS) $(ZLIB_FLAGS) $(INCLUDES) $< -o $@

models/mlp:models/mlp.o kautodiff.o kann.o kann_data.o
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
kann_data.o: kseq.h kann_data.h
kautodiff.o: kautodiff.h
examples/rnn-bit.o: kann.h kautodiff.h
models/mlp.o: kann.h kautodiff.h kann_data.h
models/textgen.o: kann.h kautodiff.h kseq.h
