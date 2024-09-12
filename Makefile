#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = lava
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

# Add environment.yml as a dependency for requirements target
REQUIREMENTS_DEPS = environment.yml

#################################################################################
# COMMANDS                                                                      #
#################################################################################
.PHONY: clean lint format create_environment data train-vae optimize-vae train-irt compare visualize help

# Use a stamp file to determine if the conda environment needs to be updated
# Define the stamp file location
STAMP_FILE := .environment_stamp

## Install Python Dependencies
requirements: $(STAMP_FILE)


$(STAMP_FILE): environment.yml
	@echo "Setting up the environment using environment.yml"
	@conda env create --name $(PROJECT_NAME) -f environment.yml 2> /dev/null || conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	@touch $(STAMP_FILE)
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8 and black (use `make format` to do formatting)
lint:
	flake8 lava
	isort --check --diff --profile black lava
	black --check --config pyproject.toml lava


## Format source code with black
format:
	black --config pyproject.toml lava


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) lava/dataset.py


## Train the VAE model using a specific set of hyperparameters
train-vae: data
	$(PYTHON_INTERPRETER) lava/modeling/train_vae.py --mode train


## Optimize the VAE hyperparameters using Optuna
optimize-vae: data
	$(PYTHON_INTERPRETER) lava/modeling/train_vae.py --mode optimize


## Train the IRT model
train-irt: data
	$(PYTHON_INTERPRETER) lava/modeling/train_irt.py


## Compare the performance of the IRT and VAE models
compare: data
	$(PYTHON_INTERPRETER) lava/modeling/compare_models.py


visualize: data
	jupyter nbconvert --to notebook --execute notebooks/01_interpret_vae_latent_space.ipynb

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
