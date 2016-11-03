CC=			gcc
CFLAGS=		-g -Wall -Wc++-compat -O2
CPPFLAGS=
ZLIB_FLAGS=	-DHAVE_ZLIB   # comment out this line to drop the zlib dependency
INCLUDES=	-I.
OBJS=		kautodiff.o kann_rand.o kann_data.o ann.o model.o
PROG=		kann
LIBS=		-lm -lz

.SUFFIXES:.c .o
.PHONY:all demo clean depend

.c.o:
		$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

all:libkann.a $(PROG)

kann:libkann.a cli.o main.o
		$(CC) cli.o main.o -o $@ -L. -lkann $(LIBS)

libkann.a:$(OBJS)
		$(AR) -csru $@ $(OBJS)

kann_data.o:kann_data.c
		$(CC) -c $(CFLAGS) $(ZLIB_FLAGS) $(INCLUDES) $< -o $@

clean:
		rm -fr gmon.out *.o a.out $(PROG) *~ *.a *.dSYM

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c)

# DO NOT DELETE

ann.o: kann_rand.h kann.h kautodiff.h
cli.o: kann.h kautodiff.h kann_data.h
kann_data.o: kseq.h kann_data.h
kann_rand.o: kann_rand.h
kautodiff.o: kautodiff.h
main.o: kann.h kautodiff.h
model.o: kann_rand.h kann.h kautodiff.h
