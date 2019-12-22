PROGRAM=game_of_life_mpi_cols

.PHONY: build
build: $(PROGRAM)
	echo "building $(PROGRAM)"

$(PROGRAM): $(PROGRAM).c
	mpicc -g -Wall -o $(PROGRAM) $(PROGRAM).c

.PHONY: run
run:
	@for num in `seq 1 60`; do \
		mpirun -n 8 --hostfile ../../hostfile ./$(PROGRAM) input$$num; \
		mpirun -n 8 --hostfile ../../hostfile ./$(PROGRAM) input$$num; \
		mpirun -n 8 --hostfile ../../hostfile ./$(PROGRAM) input$$num; \
		mpirun -n 8 --hostfile ../../hostfile ./$(PROGRAM) input$$num; \
		mpirun -n 8 --hostfile ../../hostfile ./$(PROGRAM) input$$num; \
		mpirun -n 8 --hostfile ../../hostfile ./$(PROGRAM) input$$num; \
		mpirun -n 8 --hostfile ../../hostfile ./$(PROGRAM) input$$num; \
		mpirun -n 8 --hostfile ../../hostfile ./$(PROGRAM) input$$num; \
		mpirun -n 8 --hostfile ../../hostfile ./$(PROGRAM) input$$num; \
		mpirun -n 8 --hostfile ../../hostfile ./$(PROGRAM) input$$num; \
	done;

.PHONY: clean
clean:
	rm output* timing*

