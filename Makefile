CXX      = g++ -std=gnu++11
CXXFLAGS = -O3 -Wall -g

ALL = binarize binarize-sauvola isolate-bg mask-op

all: $(ALL)
clean:
	rm -f $(ALL) *.o

.cpp.o:
	$(CXX) -c -o $@ $(CXXFLAGS) $<

OBJ_BINARIZE = \
	binarize.o \
	microlib-argparse.o
OBJ_BINARIZE_SAUVOLA = \
	binarize-sauvola.o \
	microlib-argparse.o
OBJ_ISOLATE_BG = \
	isolate-bg.o \
	microlib-argparse.o
OBJ_MASK_OP = \
	mask-op.o \
	microlib-argparse.o

binarize.o: microlib/argparse.hpp
binarize-sauvola.o: microlib/argparse.hpp
isolate-bg.o: microlib/argparse.hpp
mask-op.o: microlib/argparse.hpp

microlib-argparse.o: microlib/argparse.hpp

binarize: $(OBJ_BINARIZE)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJ_BINARIZE) -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
binarize-sauvola: $(OBJ_BINARIZE_SAUVOLA)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJ_BINARIZE_SAUVOLA) -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
isolate-bg: $(OBJ_ISOLATE_BG)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJ_ISOLATE_BG) -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
mask-op: $(OBJ_MASK_OP)
	$(CXX) -o $@ $(CXXFLAGS) $(OBJ_MASK_OP) -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

.PHONY: all clean
