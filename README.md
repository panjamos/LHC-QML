# LHC-QML
## Running the code
1. Install conda if you have not already (this makes managing the environment and packages easier)  
  [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [miniforge](https://github.com/conda-forge/miniforge) should be fine
3. Create a directory on your local computer and go into it
4. Use git to clone this repo i.e.  
	`git clone https://github.com/panjamos/LHC-QML.git`
4. Create conda environment using environment.yml to install all necessary packages  
	`conda env create - f environment.yml -n [NAME]`
5. Activate conda environment  
	`conda activate [NAME]`
6. Run code  
	`python train.py`
