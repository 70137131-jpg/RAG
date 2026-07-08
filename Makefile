.PHONY: help install ingest run dev test clean

PYTHON ?= python3
SAMPLES ?= 100
CONFIG ?= balanced
PORT ?= 5000

help:
	@echo "Targets:"
	@echo "  make install   Install Python dependencies"
	@echo "  make ingest    Build the vector index ($(SAMPLES) SQuAD samples, $(CONFIG) config)"
	@echo "  make run       Start the web server on port $(PORT)"
	@echo "  make dev       Start the server with Flask debug mode"
	@echo "  make test      Run the test suite"
	@echo "  make clean     Remove the vector store and caches"

install:
	$(PYTHON) -m pip install -r requirements.txt

ingest:
	$(PYTHON) ingest.py --config $(CONFIG) --samples $(SAMPLES) --yes

run:
	PORT=$(PORT) $(PYTHON) app.py

dev:
	FLASK_ENV=development PORT=$(PORT) $(PYTHON) app.py

test:
	$(PYTHON) -m pytest

clean:
	rm -rf chroma_db __pycache__ tests/__pycache__ .pytest_cache evaluation_results.json
