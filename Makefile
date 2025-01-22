scan: main.cu knogge_stone_scan.o Makefile profile_function.h
	nvcc -o scan main.cu knogge_stone_scan.o

knogge_stone_scan.o: knogge_stone_scan.cu
	nvcc -c knogge_stone_scan.cu
