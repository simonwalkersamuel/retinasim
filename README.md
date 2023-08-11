# RetinaSim

**Retinal layer and vasculature simulation**

<center><img src="https://github.com/simonwalkersamuel/retinasim/assets/21674318/9ada423d-edcf-4df8-9d9d-e35c9b7150f0)" width="300" height="300" /></center>

## Paper

Physics-informed deep generative learning for quantitative assessment of the retina

Emmeline Brown, Andrew Guy, Natalie Holroyd, Paul Sweeney, Lucie Gourmet, Hannah Coleman, Claire Walsh, Rebecca Shipley, Ranjan Rajendram, Simon Walker-Samuel

bioRxiv 2023.07.10.548427; doi: https://doi.org/10.1101/2023.07.10.548427

## License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

## Installation

RetinaSim relies on several libraries:
1) The code in this repository (RetinaSim) is written in python (3.8), and both provides functionality and glues together the other libraries;
2) Reanimate for 1D flow simulation (https://www.psweeney.co.uk/research/reanimate);
3) Vascular.Networks for procedural modelling of blood vessel networks (https://github.com/AndrewAGuy/vascular-networks);
4) Pymira for creating and editing spatial graph structures in python

RetinaSim is under development and has so far only been installed and tested on Ubuntu (20.04.5)

### Installing RetinaSim

1) Install anaconda and create a virtual environment containing python 3.8.
2) Start the virtual environment and enter the `site-packages` folder
3) Clone this repository (by default will be cloned into a folder named retinasim)
4) Install dependencies in requirements.txt usng pip

### Installing Reanimate

1) In the virtual environment, cd to the site-packages folder
2) Clone https://github.com/psweens/Reanimate/tree/retinasim/Reanimate
3) Switch to the retinasim branch of the Reanimate repository
4) Compile the c++ code
5) Cd to the retinasim directory (created above) and edit the `config.py` directory. REANIMATE_DIR should be changed to the location of the cloned repository (with an additional 'Reanimate' directory appended to the end - e.g. /VIRTUAL_ENV_PATH/site-packages/Reanimate/Reanimate)
6) Also edit REANIMATE_TMP_DIR_LOC to correpsond to a folder for temporary storage results during processing (deleted at completion)

### Installing Vascular.Networks
1) Follow instructions on https://github.com/AndrewAGuy/vascular-networks

## Installing Pymira
1) Clone https://github.com/CABI-SWS/pymira into the site-packages folder in the virtual environment

## Running RetinaSim
From a command line interface, activate the virtual environment and cd to site-packages/retinasim.

Enter `python main.py /OUTPUT/DIRECTORY/` for default operation

## Link to data
Example simulation data referenced in the paper can be found here:
https://www.dropbox.com/scl/fo/whwru5rmz8g7cr0h8ytg1/h?rlkey=ynbh2kdhe0pcvpfo6cypm9oc6&dl=0
Example retinal vessel segmentations referenced in the paper can be found here:
https://www.dropbox.com/scl/fo/nambm5mg4434aq7zmm3vs/h?rlkey=4kmkof44xptxir6h30xno01ia&dl=0
