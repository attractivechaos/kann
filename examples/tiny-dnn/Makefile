CC=			gcc
CFLAGS=		-g -Wall -Wc++-compat -O2
CXX=		g++
CXXFLAGS=	-g -Wall -O3 -std=c++11 -ffast-math
CPPFLAGS=	-DCNN_SINGLE_THREAD
INCLUDES=	-I../..
EXE=		mnist-mlp mlp
LIBS=		-lm -lz

ifdef TINY_DNN
	INCLUDES+=-I$(TINY_DNN)
endif

.SUFFIXES:.c .cpp .o
.PHONY:all clean

.c.o:
		$(CC) -c $(CFLAGS) $(INCLUDES) $(CPPFLAGS) $< -o $@

.cpp.o:
		$(CXX) -c $(CXXFLAGS) $(INCLUDES) $(CPPFLAGS) $< -o $@

all:$(EXE)

../../kann_extra/kann_data.o:../../kann_extra/kann_data.c
		$(CC) -c $(CFLAGS) -DHAVE_ZLIB $< -o $@

mlp:mlp.o ../../kann_extra/kann_data.o
		$(CXX) -o $@ $^ $(LIBS)

mnist-mlp:mnist-mlp.o
		$(CXX) -o $@ $^ $(LIBS)

clean:
		rm -fr *.o */*.o a.out */a.out *.a *.dSYM */*.dSYM $(EXE)
