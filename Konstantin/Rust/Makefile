VERSIONS := \
	game_of_life \
	game_of_life_cols

.PHONY: build
build:
	@for d in $(VERSIONS); do \
		$(MAKE) -s -C $$d build; \
	done;

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

