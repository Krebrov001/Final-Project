PROGRAM=game_of_life_mpi_cols

.PHONY: run
run:
	@for num in `seq 1 60`; do \
		mpirun -n 8 --hostfile ../../hostfile python3 $(PROGRAM).py input$$num; \
		mpirun -n 8 --hostfile ../../hostfile python3 $(PROGRAM).py input$$num; \
		mpirun -n 8 --hostfile ../../hostfile python3 $(PROGRAM).py input$$num; \
		mpirun -n 8 --hostfile ../../hostfile python3 $(PROGRAM).py input$$num; \
		mpirun -n 8 --hostfile ../../hostfile python3 $(PROGRAM).py input$$num; \
		mpirun -n 8 --hostfile ../../hostfile python3 $(PROGRAM).py input$$num; \
		mpirun -n 8 --hostfile ../../hostfile python3 $(PROGRAM).py input$$num; \
		mpirun -n 8 --hostfile ../../hostfile python3 $(PROGRAM).py input$$num; \
		mpirun -n 8 --hostfile ../../hostfile python3 $(PROGRAM).py input$$num; \
		mpirun -n 8 --hostfile ../../hostfile python3 $(PROGRAM).py input$$num; \
	done;

.PHONY: clean
clean:
	rm output* timing*

