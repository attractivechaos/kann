CC=			gcc
CFLAGS=		-g -Wall -Wc++-compat -O2
CPPFLAGS=
ZLIB_FLAGS=	-DHAVE_ZLIB   # comment out this line to drop the zlib dependency
INCLUDES=	-I.
OBJS=		kautodiff.o kann_rand.o kann_data.o kann_ann.o kann_model.o
PROG=
LIBS=		-lm -lz

.SUFFIXES:.c .o
.PHONY:all demo clean depend

.c.o:
		$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

all:libkann.a

libkann.a:$(OBJS)
		$(AR) -csru $@ $(OBJS)

kann_data.o:kann_data.c
		$(CC) -c $(CFLAGS) $(ZLIB_FLAGS) $(INCLUDES) $< -o $@

clean:
		rm -fr gmon.out *.o a.out $(PROG) *~ *.a *.dSYM

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c)

# DO NOT DELETE

kann_ann.o: kann_rand.h kann_ann.h kautodiff.h
kann_data.o: kseq.h kann_data.h
kann_mlp.o: kann_rand.h kann_ann.h kautodiff.h
kann_rand.o: kann_rand.h
kautodiff.o: kautodiff.h
test.o: kautodiff.h
