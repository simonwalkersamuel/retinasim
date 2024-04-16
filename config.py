import os
ROOT_DIR = os.path.dirname(__file__)

# REANIMATE (flow estimation) directories
REANIMATE_DIR = os.path.join(ROOT_DIR,"Reanimate/Reanimate")
REANIMATE_TMP_DIR_LOC = os.path.join(ROOT_DIR,"temp")

# RetinaGen (vessel simulation) directories
RETINAGEN_DIR = os.path.join(ROOT_DIR,"RetinaGen/RetinaGen/bin/Debug/net6.0")
RETINAGEN_DATA_DIR = os.path.join(ROOT_DIR,"data/RetinaGen")
