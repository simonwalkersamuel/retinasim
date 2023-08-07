
# REANIMATE (flow estimation) directories
REANIMATE_DIR = "/PATH/TO/REANIMATE/DISTRIBUTION" 
#REANIMATE_DATA_FILENAME = "/PATH/TO/REANIMATE/DAT_FILE.dat"
REANIMATE_TMP_DIR_LOC = "/PATH/TO/REANIMATE/TEMP/DIRECTORY"

import os
if os.name=='nt':
    REANIMATE_DIR = r"C:\Users\simon\Reanimate\Reanimate"
    #REANIMATE_DATA_FILENAME = r"C:\Users\simon\desktop\cco\retina_cco.dat"
    REANIMATE_TMP_DIR_LOC = r"C:\Users\simon\desktop\temp"
else:
    REANIMATE_DIR = "/mnt/ml/anaconda_envs/vessel_growth_38/lib/python3.8/site-packages/Reanimate/Reanimate"
    #REANIMATE_DATA_FILENAME = "/mnt/data2/retinasim/cco/graph/retina_cco.dat"
    REANIMATE_TMP_DIR_LOC = "/mnt/data2/temp"
    
    O2_DIR = "/mnt/ml/anaconda_envs/vessel_growth_38/lib/python3.8/site-packages/TWS_Steady_Greens_O2/build-dir/Tim_O2"
    O2_DATA_FILENAME = "/mnt/data2/retinasim/cco/graph/retina_cco.dat"
    O2_TMP_DIR_LOC = "/mnt/data2/temp"
