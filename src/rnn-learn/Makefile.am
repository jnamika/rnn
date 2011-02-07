bin_PROGRAMS = rnn-learn
rnn_learn_SOURCES = main.c target.c training.c print.c parse.c ../common/rnn.c ../common/rnn_lyapunov.c ../common/entropy.c ../common/solver.c ../common/utils.c
AM_CPPFLAGS = -I ../common -DENABLE_ADAPTIVE_LEARNING_RATE -DENABLE_ATTRACTION_OF_INIT_C -D_POSIX_C_SOURCE=200112L
AM_CFLAGS = $(OPENMP_CFLAGS)
