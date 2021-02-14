doStuff:
	g++ -fopenmp -o omp omp.cpp
	nvcc -g -Xcompiler -fopenmp -o cuda cuda.cu -lm -lgomp

clean:
	rm -f omp cuda
