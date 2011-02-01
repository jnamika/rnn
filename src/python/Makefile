# Makefile for: python scripts
# Data created: 4 Jan 2011
# Whom: Jun Namikawa

TARGET = librunner.so
VERSION = 1

CC = gcc
CPP = gcc

OBJ_DIR = ../../obj/python
BIN_DIR = ../../bin
COMMON_DIR = ../common

INCLUDE_DIR := -I. -I$(COMMON_DIR)
LIB_DIR :=
WARNFLAG := -Wall -Wextra -Wno-unused-parameter
STD := -std=gnu99 -pedantic
OPT := -O3 -fPIC
MACROS :=
SYSLIBS := -lm
DEBUG :=

CFLAGS = $(WARNFLAG) $(STD) $(OPT) $(MACROS) $(INCLUDE_DIR) $(LIB_DIR) $(DEBUG)


vpath %.c $(COMMON_DIR)
SRCS = rnn_runner.c rnn.c utils.c
PY_SRCS = rnn_runner.py print_log.py plot_log.py gen_target.py rnn_scale.py rnn_kl_div.py
SH_SRCS = print-log plot-log rnn-scale rnn-scale-restore rnn-kl-div
OBJS = $(SRCS:%.c=$(OBJ_DIR)/%.o)


.PHONY: all
all: depend.mk $(OBJ_DIR) $(BIN_DIR) $(BIN_DIR)/$(TARGET) $(PY_SRCS) $(SH_SRCS)
	-@ cp $(PY_SRCS) $(SH_SRCS) $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/$(TARGET): $(OBJS)
	$(CC) -shared -o $@ $(OBJS) $(CFLAGS) $(SYSLIBS)

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
	-@ rm -rf $(OBJ_DIR) $(BIN_DIR)/$(TARGET) $(PY_SRCS:%=$(BIN_DIR)/%) $(SH_SRCS:%=$(BIN_DIR)/%) depend.mk

.PHONY: help
help: info

info:
	-@ echo
	-@ echo "Makefile for shared library of recurrent neural networks used by python script"
	-@ echo "Jun Namikawa, 2010"
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

