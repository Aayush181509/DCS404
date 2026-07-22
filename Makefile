# Makefile for Python and MkDocs commands
# Use bash for shell commands
SHELL := /bin/bash
# Default variables
INPUT_DIR  := notebooks


NOTEBOOK ?= notebooks/02_ml_workflow.ipynb
# Default output directory for converted notebooks
# If OUTPUT is not set, it defaults to docs/lectures/
OUTPUT   ?= docs/DCS404/
OUTPUT_DIR := docs/DCS404

NOTEBOOKS  := $(wildcard $(INPUT_DIR)/*.ipynb)

.PHONY: run deploy build serve convert all convert-all convert-project

run:
	python main.py

deploy:
	mkdocs gh-deploy

build:
	mkdocs build

serve:
	mkdocs serve

convert:
	jupyter nbconvert --to markdown $(NOTEBOOK) --output-dir=$(OUTPUT)

# Batch convert all notebooks
convert-all:
	@echo "Converting all notebooks in $(INPUT_DIR)/ to Markdown..."
	@mkdir -p $(OUTPUT_DIR)
	@for nb in $(NOTEBOOKS); do \
		echo "Converting $$nb..."; \
		jupyter nbconvert --to markdown $$nb --output-dir=$(OUTPUT_DIR); \
	done

# Final project notebook (lives in notebooks/project/, not picked up by convert-all)
convert-project:
	jupyter nbconvert --to markdown notebooks/project/00_final_project.ipynb --output-dir=$(OUTPUT_DIR)/project

all: build serve
