PROGRAM=target/debug/game_of_life

.PHONY: build
build: $(PROGRAM)
	echo "building $(PROGRAM)"

$(PROGRAM): src/main.rs
	cargo build

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

