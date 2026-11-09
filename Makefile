BUILD_DIR ?= build
CMAKE ?= cmake

.PHONY: all clean

all:
	$(CMAKE) -S . -B $(BUILD_DIR)
	$(CMAKE) --build $(BUILD_DIR)

clean:
	@if [ -d "$(BUILD_DIR)" ]; then $(CMAKE) --build "$(BUILD_DIR)" --target clean; fi
	$(CMAKE) -E rm -f main
