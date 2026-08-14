
.PHONY: clean_dist create_dist upload_package setup create_block_mask install_create_block_mask test_create_block_mask test_arbitrary_mask_varlen

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
CUTLASS_DSL_PACKAGE ?= nvidia-cutlass-dsl>=4.4.2
CUTE_DSL_DEPS ?= \
	"$(CUTLASS_DSL_PACKAGE)" \
	torch \
	einops \
	typing_extensions \
	"apache-tvm-ffi>=0.1.5,<0.2" \
	torch-c-dlpack-ext \
	"quack-kernels>=0.4.0"
CUTE_DSL_TEST_DEPS ?= pytest pytest-xdist ruff

clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	python setup.py sdist

upload_package: create_dist
	twine upload dist/*

setup:
	$(PIP) install $(CUTE_DSL_DEPS) $(CUTE_DSL_TEST_DEPS)
	$(MAKE) create_block_mask

create_block_mask install_create_block_mask:
	$(MAKE) -C csrc/utils create_block_mask

test_create_block_mask:
	$(MAKE) -C csrc/utils test_create_block_mask

test_arbitrary_mask_varlen:
	$(PYTHON) -m pytest tests/cute/test_arbitrary_mask_varlen.py -v
