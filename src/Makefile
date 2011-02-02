
.PHONY: all
all: 
	$(MAKE) all -C rnn-learn
	$(MAKE) all -C rnn-generate
	$(MAKE) all -C rnn-lyapunov
	$(MAKE) all -C python CC=gcc
	$(MAKE) all -C unit-test CC=gcc

.PHONY: clean
clean:
	$(MAKE) clean -C rnn-learn
	$(MAKE) clean -C rnn-generate
	$(MAKE) clean -C rnn-lyapunov
	$(MAKE) clean -C python
	$(MAKE) clean -C unit-test

