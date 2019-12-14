CXX = mpic++

C_MPI_mm: MPI_ethan_mm.o
	$(CXX) -o C_MPI_mm MPI_ethan_mm.o

MPI_ethan_mm.o: MPI_ethan_mm.cpp

clean: 
	rm -f *.o C_MPI_mm

runv: 
	mpirun -n 8 --hostfile hostfile --mca btl_tcp_if_include eth0 ./C_MPI_mm -v 8
