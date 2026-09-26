.PHONY: install test cut dry clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

dry:
	python -m loopcutter.cli cut $(MANIFEST) --out loops --dry-run

cut:
	python -m loopcutter.cli cut $(MANIFEST) --out loops

clean:
	rm -rf loops .pytest_cache **/__pycache__
