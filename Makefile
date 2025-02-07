NVCC_OPTS=

all: scan

scan: main.o knogge_stone_scan.o blelloch_scan.o
	nvcc -o scan main.o knogge_stone_scan.o blelloch_scan.o $(NVCC_OPTS)

main.o: main.cu knogge_stone_scan.h blelloch_scan.h profile_function.h check_cuda.h
	nvcc -c main.cu $(NVCC_OPTS)

knogge_stone_scan.o: knogge_stone_scan.cu knogge_stone_scan.h check_cuda.h
	nvcc -c knogge_stone_scan.cu $(NVCC_OPTS)

blelloch_scan.o: blelloch_scan.cu blelloch_scan.h check_cuda.h
	nvcc -c blelloch_scan.cu $(NVCC_OPTS)

clean:
	rm -f *.o scan