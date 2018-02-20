# Makefile for matlab-lmdb
LMDBDIR := src/liblmdb
ECHO := echo
MATLABDIR ?= /usr/local/matlab
MATLAB := $(MATLABDIR)/bin/matlab
MEX := $(MATLABDIR)/bin/mex
MEXEXT := $(shell $(MATLABDIR)/bin/mexext)
MEXFLAGS := -Iinclude -I$(LMDBDIR) CXXFLAGS="\$$CXXFLAGS -std=c++11"
TARGET := +lmdb/private/LMDB_.$(MEXEXT)

.PHONY: all test clean

all: $(TARGET)

$(TARGET): src/LMDB_.cc $(LMDBDIR)/liblmdb.a
	$(MEX) -output $@ $< $(MEXFLAGS) $(LMDBDIR)/liblmdb.a

$(LMDBDIR)/liblmdb.a: $(LMDBDIR)
	$(MAKE) -C $(LMDBDIR)

test: $(TARGET)
	$(ECHO) "run test/testLMDB" | $(MATLAB) -nodisplay

clean:
	$(MAKE) -C $(LMDBDIR) clean
	$(RM) $(TARGET)
