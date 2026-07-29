# Makefile for the storage-projections-thesis analysis pipeline.
# Common tasks: install dependencies, run the analysis, check syntax, clean.

.PHONY: help install run check clean

# Default target - print help
help:
	@echo "Available targets:"
	@echo "  make install   Install Python dependencies from requirements.txt"
	@echo "  make run       Run the full analysis pipeline"
	@echo "  make check     Syntax-check every module in src/ without running"
	@echo "  make clean     Remove all generated outputs from outputs/"
	@echo "  make help      Show this message"

install:
	pip install -r requirements.txt

# Run the analysis from outputs/ so all generated files land there.
# The pipeline expects the dataset in the current working directory;
# symlink it in (or copy if symlinks are not supported by the filesystem).
run:
	@if [ ! -f data/Energy_Storage_Data_Collection_Vyara_Tsoneva.xlsx ]; then \
		echo "ERROR: dataset not found in data/. See data/README.md."; \
		exit 1; \
	fi
	@cd outputs && \
		ln -sf ../data/Energy_Storage_Data_Collection_Vyara_Tsoneva.xlsx . && \
		python ../src/run_analysis.py

# Syntax-check every module in src/.  Runs quickly; useful after edits.
check:
	@for f in src/*.py; do \
		python -m py_compile $$f || exit 1; \
	done
	@echo "Syntax OK - all modules compile."

clean:
	@rm -f outputs/*.csv \
	       outputs/*.png \
	       outputs/*.pdf \
	       outputs/*.txt \
	       outputs/Energy_Storage_Data_Collection_Vyara_Tsoneva.xlsx
	@echo "Cleaned outputs/."