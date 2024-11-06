COBJ	= mt19937ar.o io_png.o iio.o
CXXOBJ	= libauxiliar.o libdenoising.o chambolle_ipol.o imdiff_ipol.o

BIN	= chambolle_ipol imdiff_ipol

hdrdir=-I/opt/local/include/ -I/usr/local/include/ -I/usr/include/
libdir=-L/opt/local/lib/ -L/usr/local/lib/ -L/usr/lib/

COPT	= -O3 -funroll-loops -fomit-frame-pointer  
CFLAGS  +=  $(COPT) $(hdrdir)

LDFLAGS +=  $(CXXFLAGS) $(libdir) -lpng -lm -ltiff

C=gcc
CXX=g++

default: $(COBJ) $(CXXOBJ)  $(BIN)

$(COBJ) : %.o : %.c 
	$(C) -c $(CFLAGS)   $< -o $@

$(CXXOBJ) : %.o : %.cpp 
	$(CXX) -c $(CFLAGS)   $< -o $@

$(BIN) : % : %.o  io_png.o libauxiliar.o libdenoising.o mt19937ar.o iio.o
	$(CXX)   -o $@  $^ $(LDFLAGS)

.PHONY : clean
clean:
	$(RM) $(COBJ) $(CXXOBJ) ; rm $(BIN)
