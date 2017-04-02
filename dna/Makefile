CC=			gcc
CFLAGS=		-g -Wall -Wc++-compat -O2
CPPFLAGS=
INCLUDES=	-I.. -I../kann_extra
LIBS=		-lm -lz -lpthread
EXE=		dna-cnn dna-brnn

.SUFFIXES:.c .o
.PHONY:all demo clean depend

.c.o:
		$(CC) -c $(CFLAGS) $(CPPFLAGS) $(INCLUDES) $< -o $@

all:$(EXE)

dna-cnn:../kautodiff.o ../kann.o dna-io.o dna-cnn.o
		$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

dna-brnn:../kautodiff.o ../kann.o dna-io.o dna-brnn.o
		$(CC) $(CFLAGS) $^ -o $@ $(LIBS)

clean:
		rm -fr *.o a.out *~ *.a *.dSYM $(EXE)

depend:
		(LC_ALL=C; export LC_ALL; makedepend -Y -- $(CFLAGS) $(DFLAGS) -- *.c)

# DO NOT DELETE

kann.o: ../kann.h ../kautodiff.h
kautodiff.o: ../kautodiff.h
dna-cnn.o: ../kann.h ../kautodiff.h dna-io.h
dna-io.o: dna-io.h ../kann_extra/kseq.h
dna-brnn.o: ../kann.h ../kautodiff.h dna-io.h
