SHELL=/bin/bash

SCRIPTS_DIR='scripts'
PROJECT_DIR='fireml'
CONDA_ENV_NAME=${PROJECT_DIR}
CONDA_ROOT=$$HOME/miniconda3

.PHONY: build

build_cython:
	@python ${SCRIPTS_DIR}/setup_cython.py build_ext --inplace

build: build_cython
	@python ${SCRIPTS_DIR}/setup.py develop

clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf
	@rm -f ${PROJECT_DIR}/processing/*.{so,c,html}
	@rm -f ${PROJECT_DIR}/processing/dataset/*.{so,c,html}
	@rm -f ${PROJECT_DIR}/processing/dataset/realtime/*.{so,c,html}
	@rm -f ${PROJECT_DIR}/processing/rasterization/*.{so,c,html}
	@rm -rf build/
	@rm -f .coverage
	@rm -rf .pytest_cache/
	@rm -rf ${PROJECT_DIR}.egg-info/
	@python ${SCRIPTS_DIR}/setup.py develop --uninstall
