# Makefile

VENV = myenv
PYTHON = ./$(VENV)/bin/python3
PIP = ./$(VENV)/bin/pip
JUPYTER = ./$(VENV)/bin/jupyter

install:
	python3 -m venv $(VENV)
	$(PIP) install jupyter nbconvert
	$(PIP) install -r requirements.txt

# Gaussian noise notebooks
gauss-noise:
	$(JUPYTER) nbconvert --execute --inplace photo_editing/Gauss/gauss_noise_adder.ipynb

gauss-denoise:
	$(JUPYTER) nbconvert --execute --inplace photo_editing/Gauss/denoise.ipynb

# Poisson noise notebook
poisson-denoise:
	$(JUPYTER) nbconvert --execute --inplace photo_editing/poisson/poisson_anscombe_denoising.ipynb

# Salt & Pepper and Speckle noise notebooks
sp-noise:
	$(JUPYTER) nbconvert --execute --inplace photo_editing/salt_pepper_and_speckle/noise_script.ipynb

sp-denoise:
	$(JUPYTER) nbconvert --execute --inplace photo_editing/salt_pepper_and_speckle/denoise_script.ipynb

# Data analysis notebooks
gather-gauss-data:
	$(JUPYTER) nbconvert --execute --inplace Data_Results/Data/gathering_data_gauss.ipynb

bar-graphs-gauss:
	$(JUPYTER) nbconvert --execute --inplace Data_Results/Data/bar_graphs_gauss.ipynb

# Run ALL notebooks in sequence
all: gauss-noise gauss-denoise poisson-denoise sp-noise sp-denoise gather-gauss-data bar-graphs-gauss
	@echo "✅ All notebooks executed successfully!"

clean:
	rm -rf $(VENV)

reinstall: clean install

# Help command
help:
	@echo "Available commands:"
	@echo "  make install          - Create virtual env and install dependencies"
	@echo "  make gauss-noise      - Execute Gaussian noise adder notebook"
	@echo "  make gauss-denoise    - Execute Gaussian denoising notebook"
	@echo "  make poisson-denoise  - Execute Poisson denoising notebook"
	@echo "  make sp-noise         - Execute Salt & Pepper/Speckle noise adder notebook"
	@echo "  make sp-denoise       - Execute Salt & Pepper/Speckle denoising notebook"
	@echo "  make gather-gauss-data - Execute data gathering notebook"
	@echo "  make bar-graphs-gauss - Execute bar graphs notebook"
	@echo "  make all              - Execute ALL notebooks in sequence"
	@echo "  make clean            - Remove virtual env"
	@echo "  make reinstall        - Clean and reinstall"