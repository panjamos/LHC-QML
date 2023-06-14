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
6. Modify parameters in code
7. Run code  
	`python train.py`

## Info
train.py will train and save a VQC using the parameters listed at the top of the tile. This VQC classifies datapoints as either signal events (Higgs boson detection) represented as 1 or background events (no Higgs boson detection) represented as 0. Before running, you should review and edit the parameters as necessary and ensure the datapaths are correct.  
  
plot.py will plot the discriminator and the ROC for a model given at the filepath at the beginning of the file. Before running, ensure the parameters the beginning match those of the trained model.  
  
LHC_QML_module.py contains functions as used in train.py and plot.py
