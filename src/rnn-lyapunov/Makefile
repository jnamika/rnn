# Makefile for: a program to analyze the Lyapunov exponents of recurrent neural networks
# Data created: 7 Jan 2011
# Whom: Jun Namikawa

TARGET = rnn-lyapunov
VERSION = 1.0

CC = gcc
CPP = gcc

OBJ_DIR = ../../obj/rnn-lyapunov
BIN_DIR = ../../bin
COMMON_DIR = ../common

enable_mtrace = no
enable_gprof = no


INCLUDE_DIR := -I. -I$(COMMON_DIR)
LIB_DIR :=
WARNFLAG := -Wall -Wextra -Wno-unused-parameter
STD := -std=gnu99 -pedantic
OPT := -O3 -fopenmp
MACROS := -DVERSION=$(VERSION) -DENABLE_ADAPTIVE_LEARNING_RATE -DENABLE_ATTRACTION_OF_INIT_C -DNDEBUG
SYSLIBS := -lm
DEBUG :=


ifneq ($(CC), gcc)
enable_gprof = no
endif

ifeq ($(CC), icc)
WARNFLAG := -Wall
STD := -std=c99
OPT := -O3 -openmp
MACROS := $(MACROS) -D_POSIX_C_SOURCE=200112L
endif

ifeq ($(enable_mtrace), yes)
MACROS := $(MACROS) -DENABLE_MTRACE
OPT := -O
DEBUG := $(DEBUG) -g
endif
ifeq ($(enable_gprof), yes)
DEBUG := $(DEBUG) -pg
endif


CFLAGS = $(WARNFLAG) $(STD) $(OPT) $(MACROS) $(INCLUDE_DIR) $(LIB_DIR) $(DEBUG)


vpath %.c $(COMMON_DIR)
SRCS = main.c lyapunov.c rnn.c rnn_runner.c rnn_lyapunov.c solver.c utils.c
OBJS = $(SRCS:%.c=$(OBJ_DIR)/%.o)

.PHONY: all
all: depend.mk $(OBJ_DIR) $(BIN_DIR) $(BIN_DIR)/$(TARGET)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/$(TARGET): $(OBJS)
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



.PHONY: clean
clean:
	-@ rm -rf $(OBJ_DIR) $(BIN_DIR)/$(TARGET) depend.mk

.PHONY: help
help: info

info:
	-@ echo
	-@ echo "Makefile for rnn-lyapunov"
	-@ echo "Jun Namikawa, 2011"
	-@ echo
	-@ echo "USAGE: make [all | clean | help]"
	-@ echo "all (default)      builds $(TARGET)"
	-@ echo "clean              removes target, depend.mk and .o files"
	-@ echo "help               displays help"
	-@ echo

.PHONY: echo
echo:
	@echo TARGET=$(TARGET)
	@echo CC=$(CC)
	@echo WARNFLAG=$(WARNFLAG)
	@echo STD=$(STD)
	@echo OPT=$(OPT)
	@echo MACROS=$(MACROS)
	@echo INCLUDE_DIR=$(INCLUDE_DIR)
	@echo LIB_DIR=$(LIB_DIR)
	@echo DEBUG=$(DEBUG)

