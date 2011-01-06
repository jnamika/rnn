
.PHONY: all
all: 
	$(MAKE) -C src

.PHONY: clean
clean:
	$(MAKE) clean -C src
	rm -rf obj
