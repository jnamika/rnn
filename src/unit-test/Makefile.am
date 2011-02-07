bin_PROGRAMS = rnn-unit-test
rnn_unit_test_SOURCES = main.c minunit.c test_utils.c test_rnn.c test_entropy.c test_solver.c test_rnn_lyapunov.c test_target.c test_parse.c test_rnn_runner.c ../common/rnn.c ../common/solver.c ../common/entropy.c ../common/rnn_lyapunov.c ../common/rnn_runner.c ../common/utils.c ../rnn-learn/target.c ../rnn-learn/parse.c
AM_CPPFLAGS = -I ../common -I ../rnn-learn -DENABLE_ADAPTIVE_LEARNING_RATE -DENABLE_ATTRACTION_OF_INIT_C -DMIN_VARIANCE=0.01 -D_POSIX_C_SOURCE=200112L
AM_CFLAGS = $(OPENMP_CFLAGS)
TESTS = rnn-unit-test
TESTS_ENVIRONMENT =
