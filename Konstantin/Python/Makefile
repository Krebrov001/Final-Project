VERSIONS := \
	rows \
	cols

.PHONY: run
run:
	@for d in $(VERSIONS); do \
		$(MAKE) -s -C $$d run; \
	done;

.PHONY: clean
clean:
	@for d in $(VERSIONS); do \
		$(MAKE) -s -C $$d clean; \
	done;

