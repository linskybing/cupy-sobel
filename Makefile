NVCCFLAGS  = -O3 -std=c++11 -Xptxas=-v
NVCC     = nvcc
LDFLAGS = -lpng -lz

TARGETS = sobel

.PHONY: all
all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)
