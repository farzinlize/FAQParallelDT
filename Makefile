nvcc_options= -gencode arch=compute_30,code=sm_30 --compiler-options -Wall 
sources = main.c delaunay.c

all: faq_dt

faq_dt: $(sources) Makefile
	nvcc -o build/faq_dt $(sources) $(nvcc_options)

clean:
	rm -rf build/*
