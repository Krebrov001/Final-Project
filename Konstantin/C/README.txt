 I setup my code so that the Makefiles automates the building and testing of the code. cd into the programming directory and here you will find the master Makefile and two directories rows and cols. The master Makefile completely automates the process and recursively calls the other two Makefiles in the rows and cols directories. Then type

make build

To build the mpi executables. Then please type

make run

to run the mpi executables. This might take a very long time, as the individual input files are run multiple times to obtain reliable timing information.  If you're curious, make run generates a whole bunch of output and timing files while running the mpi programs.
Please DO NOT run make clean, or make any modifications to the files other than running those two above make commands from the programming directory!

