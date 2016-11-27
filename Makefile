CC=			gcc
CFLAGS=		-g -Wall -Wc++-compat -O2
CPPFLAGS=
ZLIB_FLAGS=	-DHAVE_ZLIB   # comment out this line to drop the zlib dependency
INCLUDES=	-I.
OBJS=		kautodiff.o kad_debug.o kann_rand.o kann_min.o kann_data.o ann.o \
			layer.o reader.o
EXTRA=		models/mlp models/textgen examples/rnn-bit
LIBS=		-lm -lz

.SUFFIXES:.c .o
.PHONY:all demo clean depend

.c.o:
		$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

all:libkann.a $(EXTRA)

kann:libkann.a cli.o main.o
		$(CC) cli.o main.o -o $@ -L. -lkann $(LIBS)

libkann.a:$(OBJS)
		$(AR) -csru $@ $(OBJS)

kann_data.o:kann_data.c
		$(CC) -c $(CFLAGS) $(ZLIB_FLAGS) $(INCLUDES) $< -o $@

models/mlp:models/mlp.c libkann.a
		$(CC) $(CFLAGS) -o $@ -I. $< -L. -lkann $(LIBS)

models/textgen:models/textgen.c libkann.a
		$(CC) $(CFLAGS) -o $@ -I. $< -L. -lkann $(LIBS)

examples/rnn-bit:examples/rnn-bit.c libkann.a
		$(CC) $(CFLAGS) -o $@ -I. $< -L. -lkann $(LIBS)

examples/rnn-charnn:examples/rnn-charnn.c libkann.a
		$(CC) $(CFLAGS) -o $@ -I. $< -L. -lkann $(LIBS)

clean:
		rm -fr gmon.out *.o a.out $(EXTRA) *~ *.a *.dSYM models/*.dSYM examples/*.dSYM

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c examples/*.c models/*.c)

# DO NOT DELETE

ann.o: kann_rand.h kann_min.h kann.h kautodiff.h
kad_debug.o: kautodiff.h
kann_data.o: kseq.h kann_data.h
kann_min.o: kann_min.h
kann_rand.o: kann_rand.h
kautodiff.o: kautodiff.h
layer.o: kann_rand.h kann.h kautodiff.h
reader.o: kann_rand.h kann.h kautodiff.h
examples/rnn-bit.o: kann.h kautodiff.h kann_rand.h
models/mlp.o: kann.h kautodiff.h kann_rand.h kann_data.h
models/textgen.o: kann.h kautodiff.h kann_rand.h kseq.h
