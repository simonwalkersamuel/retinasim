# RetinaSim

**Retinal layer and vasculature simulation**

<center><img src="https://github.com/simonwalkersamuel/retinasim/assets/21674318/9ada423d-edcf-4df8-9d9d-e35c9b7150f0)" width="300" height="300" /></center>

## Paper

If you use this repository for your research, please reference:

**Physics-informed deep generative learning for quantitative assessment of the retina**

Emmeline Brown, Andrew Guy, Natalie Holroyd, Paul Sweeney, Lucie Gourmet, Hannah Coleman, Claire Walsh, Rebecca Shipley, Ranjan Rajendram, Simon Walker-Samuel

bioRxiv 2023.07.10.548427; doi: https://doi.org/10.1101/2023.07.10.548427

For more information or collaboraitons, please contact: simon.walkersamuel@ucl.ac.uk

## License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

## Installation

RetinaSim relies on several libraries:
1) The code in this repository (RetinaSim) is written in python (3.8), and both provides functionality and glues together the other libraries;
2) Reanimate for 1D flow simulation (provided here as a submodule);
3) RetinaGen for procedural modelling of blood vessel networks (provided here as a submodule);
4) Pymira for creating and editing spatial graph structures in python (provided here as a submodule)

RetinaSim is under development and has so far only been installed and tested on Ubuntu (20.04.5)

### Installing RetinaSim

1) Install anaconda (https://docs.anaconda.com/free/anaconda/install/linux/)
2) Create a virtual environment containing python 3.8.
```
conda create --name retinasim python=3.8
```
3) Start the virtual environment and enter the `site-packages` folder
```
conda activate retinasim
cd ANACONDA_DIR/envs/retinasim/lib/python3.8/site-packages
```
4) Clone this repository
```
git clone git@github.com:simonwalkersamuel/retinasim.git
```
5) Install dependencies using pip:
```
cd retinasim
pip install -r requirements.txt
```

### Installing Reanimate

1) Change directory to the Reanimate directory in this repo and compile the code (first making sure cmake is installed):
```
cd Reanimate
cmake .
make
```
2) Change directory back to the retinasim directory and edit the `config.py` directory. Edit REANIMATE_TMP_DIR_LOC to correspond to a folder for temporary storage results during processing (deleted at completion).

### Installing RetinaGen and Vascular.Networks
1) Follow instructions in the RetinaGen submodule (requires the .NET v6.0 SDK to be installed)

## Running RetinaSim
From a command line interface, activate the virtual environment and cd to retinasim.

Enter 
```
python main.py /OUTPUT/DIRECTORY/
```
for default operation

## Link to data
Example simulation data referenced in the paper can be found here:  
https://www.dropbox.com/scl/fo/whwru5rmz8g7cr0h8ytg1/h?rlkey=ynbh2kdhe0pcvpfo6cypm9oc6&dl=0  
Example retinal vessel segmentations referenced in the paper can be found here:  
https://www.dropbox.com/scl/fo/nambm5mg4434aq7zmm3vs/h?rlkey=4kmkof44xptxir6h30xno01ia&dl=0
