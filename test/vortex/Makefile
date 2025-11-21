# Vortex SIMT Backend Tests Makefile

CC = gcc
CXX = g++
CFLAGS = -Wall -O2 -I../../runtime/include -I../../runtime
CXXFLAGS = $(CFLAGS)
LDFLAGS = -lm -lpthread

# Optional OpenMP support
ifdef USE_OPENMP
	CFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif

# Test binaries
TESTS = test_vector_add \
        test_reduction \
        test_vortex_backend

# Backend binary
BACKEND = ../../build/runtime/bc_riscv

.PHONY: all clean test run-tests

all: $(TESTS)

# Compile test binaries
test_vector_add: test_vector_add.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test_reduction: test_reduction.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

test_vortex_backend: test_vortex_backend.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# Run all tests
test: all
	@echo "================================"
	@echo "Running Vortex SIMT Tests"
	@echo "================================"
	@for test in $(TESTS); do \
		echo "\n>>> Running $$test..."; \
		./$$test || echo "FAILED: $$test"; \
	done
	@echo "\n================================"
	@echo "Tests Complete"
	@echo "================================"

# Run with verbose output
test-verbose: all
	@for test in $(TESTS); do \
		echo "\n=== $$test ==="; \
		./$$test; \
		echo "Exit code: $$?"; \
	done

# Quick sanity check
quick-test: test_vortex_backend
	@echo "Running quick sanity check..."
	@./test_vortex_backend

# Clean build artifacts
clean:
	rm -f $(TESTS)
	rm -f *.o
	rm -f *.log
	rm -f *.trace

# Help
help:
	@echo "Vortex SIMT Tests Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all tests"
	@echo "  test         - Build and run all tests"
	@echo "  quick-test   - Run quick sanity check"
	@echo "  clean        - Remove build artifacts"
	@echo ""
	@echo "Options:"
	@echo "  USE_OPENMP=1 - Enable OpenMP support"
	@echo ""
	@echo "Examples:"
	@echo "  make all                    # Build all tests"
	@echo "  make test                   # Run all tests"
	@echo "  make USE_OPENMP=1 test      # Run with OpenMP"
