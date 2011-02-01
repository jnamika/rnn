# Makefile for: unit test
# Data created: 3 Jan 2011
# Whom: Jun Namikawa

TARGET = unit-test
VERSION = 1.0

CC = gcc
CPP = gcc

OBJ_DIR = ../../obj/unit-test
COMMON_DIR = ../common
RNN_LEARN_DIR = ../rnn-learn

enable_mtrace = no

INCLUDE_DIR := -I. -I$(COMMON_DIR) -I$(RNN_LEARN_DIR)
LIB_DIR :=
WARNFLAG := -Wall -Wextra
STD := -std=c99 -pedantic
OPT := -O -fopenmp -fprofile-arcs -ftest-coverage -fstack-check
MACROS := -DHAVE_CONFIG_H -DENABLE_ADAPTIVE_LEARNING_RATE -DENABLE_ATTRACTION_OF_INIT_C -DMIN_VARIANCE=0.01 -D_POSIX_C_SOURCE=200112L
SYSLIBS := -lm
DEBUG := -g

ifeq ($(enable_mtrace), yes)
MACROS := $(MACROS) -DENABLE_MTRACE
endif


CFLAGS = $(WARNFLAG) $(STD) $(OPT) $(MACROS) $(INCLUDE_DIR) $(LIB_DIR) $(DEBUG)


vpath %.c $(COMMON_DIR) $(RNN_LEARN_DIR)
SRCS = main.c minunit.c test_utils.c test_rnn.c test_entropy.c test_solver.c test_rnn_lyapunov.c test_target.c test_parse.c test_rnn_runner.c rnn.c solver.c entropy.c rnn_lyapunov.c target.c rnn_runner.c parse.c utils.c
OBJS = $(SRCS:%.c=$(OBJ_DIR)/%.o)

.PHONY: all
all: depend.mk $(OBJ_DIR) $(OBJ_DIR)/$(TARGET)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(CFLAGS) $(SYSLIBS)

$(OBJ_DIR)/%.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS)


depend.mk: $(SRCS)
	-@ echo "creating depend.mk"
	-@ rm -f depend.mk
	-@ for i in $^ ; do \
		($(CPP) -MM $(INCLUDE_DIR) $(MACROS) $$i >> depend.mk); \
	done
	-@ cat depend.mk | sed 's:^.*\.o:$(OBJ_DIR)/&:g' > .depend.mk
	-@ mv .depend.mk depend.mk

-include depend.mk


.PHONY: test
test: depend.mk $(OBJ_DIR) $(OBJ_DIR)/$(TARGET)
	-@ $(OBJ_DIR)/$(TARGET)
	-@ rm -f coverage.log
	-@ for i in $(SRCS); do (gcov -n -b -o $(OBJ_DIR) $$i >> coverage.log); done
	-@ rm -f $(OBJ_DIR)/*.gcda

.PHONY: clean
clean:
	-@ rm -rf $(OBJ_DIR) depend.mk

.PHONY: help
help: info

info:
	-@ echo
	-@ echo "Makefile for unit test"
	-@ echo "Jun Namikawa, 2011"
	-@ echo
	-@ echo "USAGE: make [all | test | clean | help]"
	-@ echo "all (default)      builds $(TARGET)"
	-@ echo "test               builds $(TARGET) and runs test"
	-@ echo "clean              removes target, depend.mk and .o files"
	-@ echo "help               displays help"
	-@ echo

