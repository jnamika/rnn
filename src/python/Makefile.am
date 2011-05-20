lib_LTLIBRARIES = librnnrunner.la
librnnrunner_la_SOURCES = ../common/rnn.c ../common/rnn_runner.c ../common/utils.c
AM_LDFLAGS = -version-info 0:0:0
AM_CPPFLAGS = -I ../common -DENABLE_ADAPTIVE_LEARNING_RATE -DENABLE_ATTRACTION_OF_INIT_C -D_POSIX_C_SOURCE=200112L
PY_SRCS = rnn_print_log.py rnn_plot_log.py rnn_scale.py rnn_runner.py rnn_kl_div.py rnn_generate_with_file.py
SH_SRCS = rnn-print-log rnn-plot-log rnn-scale rnn-scale-restore rnn-kl-div rnn-generate-with-file
bin_SCRIPTS = $(PY_SRCS) $(SH_SRCS)
