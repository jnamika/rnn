bin_PROGRAMS = rnn-lyapunov
rnn_lyapunov_SOURCES = main.c lyapunov.c ../common/rnn.c ../common/rnn_runner.c ../common/rnn_lyapunov.c ../common/solver.c ../common/utils.c
AM_CPPFLAGS = -I ../common -DENABLE_ADAPTIVE_LEARNING_RATE -DENABLE_ATTRACTION_OF_INIT_C -D_POSIX_C_SOURCE=200112L
AM_CFLAGS = $(OPENMP_CFLAGS)
