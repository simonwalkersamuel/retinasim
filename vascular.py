import os
join = os.path.join
from pymira.amirajson import convert
from pymira.csv2amira import csv2amira
from pymira import combine_graphs
import subprocess
import shutil
import time
from pathlib import Path
import numpy as np
import json

from retinasim.utility import *

PATH = RETINAGEN_DIR
DATA_PATH = RETINAGEN_DATA_DIR
OPATH = DATA_PATH

if os.name=='nt':   
    EXE_PATH = join(PATH,'RetinaGen.exe')
else:
    EXE_PATH = join(PATH,'RetinaGen')

class VascularConfig(object):

    """ Object class for reading and writing configuration files """

    def __init__(self,eye,**kwargs):
    
        self.win_proc = False
        
        if eye is None:
            from vessel_sim.retina_cco.retina_main import Eye
            eye = Eye()
    
        self.eye = eye
        self.dataPath = '' #r'C:\\Users\\simon\\vascular_networks\\vascular-ucl-collab\\Retina\\data'
        self.version = 2
        
        if self.version==1:
            self.params = {
                              "ArteryInPath": "",
                              "VeinInPath": "",
                              "ArteryOutPath": "",
                              "VeinOutPath": "",
                              "LengthUnitsPerMetre": 1000000,
                              "OpticDiscPosition": [eye.optic_disc_centre[0],eye.optic_disc_centre[1],0.],
                              "RootOffset": [2, 0, 0],
                              "FoveaPosition": [eye.fovea_centre[0],eye.fovea_centre[1],eye.fovea_centre[2]],
                              "DomainAABB": [ [-5000, 15000 ], [-6000, 0] ],
                              "MaculaRadius": eye.macula_radius,
                              "FoveaRadius": eye.fovea_radius,
                              "LatticeType": "H",
                              "VenousOffset": -0.5,
                              "DomainRadius": np.min((eye.domain[:,1]-eye.domain[:,0])[0:2]/2.),
                              "DomainCentre": [eye.optic_disc_centre[0],eye.optic_disc_centre[1],0.],
                              "SemiCircle": 1,
                              "RebalanceQ": 2,
                              "RebalanceR": 4,
                              "CoarseRebalanceIterations": 5,
                              "PreCoarseSpacing": 4000,
                              "CoarseSpacingMax": 2500,
                              "CoarseSpacingMin": 300,
                              "CoarseRefinements": 5,
                              "AsymmetryLimit": 5,
                              "SchreinerCosts": [ [1, 1, 1 ] ],
                              "PumpingWorkFactor": 0,
                              "Viscosity": 0.004,
                              "FlowRateDensity": 0.016666666666666666,
                              "MaculaFlowFactor": 4,
                              "MaculaPerturb": True,
                              "MaculaSpacingMax": eye.macula_radius/4.,
                              "MaculaSpacingMin": 150,
                              "MaculaRefinements": 2,
                              "MaculaIterations": 0,
                              "ProjectToSurface": False,
                              "SurfaceQuadraticCoefficients": [0,0,0,0,0,0]
                            }
            self.margin = self.params['CoarseSpacingMin']
        elif self.version==2:
            self.params = {
                              "ArteryInPath": "",
                              "VeinInPath": "",
                              "ArteryOutPath": "",
                              "VeinOutPath": "",
                              "Actions": "coarse;macula;resolve",
                              "Seed": -1,
                              "Domain": {
                                "LengthUnitsPerMetre": 1000000,
                                "OpticDiscPosition": [eye.optic_disc_centre[0],eye.optic_disc_centre[1],0.],
                                "FoveaPosition": [eye.fovea_centre[0],eye.fovea_centre[1],eye.fovea_centre[2]],
                                "MaculaRadius": eye.macula_radius,
                                "FoveaRadius": eye.fovea_radius,
                                "DomainRadius": np.min((eye.domain[:,1]-eye.domain[:,0])[0:2]/2.),
                                "DomainCentre": [eye.optic_disc_centre[0],eye.optic_disc_centre[1],0.],
                                "SemiCircle": 1,
                                "Viscosity": 0.004,
                                "FlowRateDensity": 0.0167,
                                "ArteryMurrayExponent": 2.5,
                                "VeinMurrayExponent": 2.5
                              },
                              "Major": {
                                "SpacingMax": 2500.,
                                "SpacingMin": 150.,
                                "Refinements": 5,
                                "RemoveStartingLeaves": True,
                                "PreSpacing": 0,
                                "ClearMacula": True,
                                "TerminalDeviation": 0,
                                "TerminalsMobile": False,
                                "FrozenFactor": 0,
                                "UpdateFrozen": False,
                                "FreezeBranches": False
                              },
                              "Macula": {
                                "FlowFactor": 1,
                                "SpacingMax": eye.macula_radius/4.,
                                "SpacingMin": 150,
                                "Refinements": 5,
                                "Sparsity": 0.5,
                                "MobileRangeFactor": 1,
                                "OptimizeTopology": False,
                                "MinAlignment": -0.7
                              },
                              "Interleaving": {
                                "Radius": 5,
                                "PaddingFactor": 1.25,
                                "BifurcatingFlow": "Infinity",
                                "FrozenFlow": "Infinity",
                                "Iterations": 5,
                                "Macula": False
                              },
                              "Optimizer": {
                                "MaxRadiusAsymmetry": 5,
                                "BranchShortFraction": 0.1,
                                "Frozen": None,
                                "MinTerminalLength": 0,
                                "TargetStep": 0
                              }
                            }
        
        self.graph_artery_filename = ''
        self.win_proc = False
        self.quad = 'upper'
        
        for k,v in zip(kwargs.keys(),kwargs.values()):
            if hasattr(self,k):
                setattr(self,k,v)
            elif k in self.params.keys():
                self.params[k] = v
            else:
                print(f'Warning, attribute not recognised! {k}')
        
        self.set_quad()
        
    def set_quad(self,quad=None):
    
        if quad is not None:
            self.quad = quad
            
        if self.quad=='upper':
            self.params["Domain"]["SemiCircle"] = -1
        elif self.quad=='lower':
            self.params["Domain"]["SemiCircle"] = 1 
        else:
            print(f'Error, quadrant value not recognised: {self.quad}')        
    
    def write(self,ofile=None,win_proc=False,set_quad=True,regulate_filepaths=True):
    
        if self.win_proc:
            pathdiv = r'\\'
        else:
            pathdiv = '/'
          
        if set_quad==True:  
            self.set_quad()
            
        graph_fname = Path(self.graph_artery_filename).stem
        if self.params['ArteryInPath']=='' and regulate_filepaths:
            self.params['ArteryInPath'] = self.dataPath + pathdiv + graph_fname + '.json'
        if self.params['VeinInPath']=='' and regulate_filepaths:
            self.params['VeinInPath'] = self.dataPath + pathdiv + graph_fname.replace('artery','vein') + '.json'
        if self.params['ArteryOutPath']=='' and regulate_filepaths:
            self.params['ArteryOutPath'] = self.dataPath + pathdiv + graph_fname + '_cco.csv'
        if self.params['VeinOutPath']=='' and regulate_filepaths:
            self.params['VeinOutPath'] = self.dataPath + pathdiv + graph_fname.replace('artery','vein') + '_cco.csv'  

        with open(ofile,'w') as fh:
            json.dump(self.params,fh,indent=4)
                    
    def read(self,fname):
        with open(fname, 'r') as f:
            data = json.load(f)
        self.params = data
                            
    def print(self):
        for k,v in zip(self.params.keys(),self.params.values()):
            if type(v) is dict:
                print(f'{k}:')
                for k1,v1 in zip(v.keys(),v.values()):
                    if type(v1) is list and len(v1)>3:
                        print(f'   {k1}: {v1[:3]}...')
                    else:
                        print(f'   {k1}: {v1}')
            else:
                print(f'{k}: {v}')
def run_sim(fname,exe_path=EXE_PATH,quiet=True):
    cmd = [exe_path, fname]
    print(f'Simulating from {fname}')

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if quiet==False:
        for line in p.stdout:
            print(line)
    p.wait()
    if quiet==False:
        print(p.returncode)
    return p.returncode

def run_init(exe_path=EXE_PATH):
    print(f'Initialising config file: {exe_path}')
    cmd = [exe_path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    for line in p.stdout:
        print(line)
    p.wait()

    print(p.returncode)

def simulate(convert_to_json=True,simulate=True,combine=True,exe_path=EXE_PATH,path=PATH,data_path=DATA_PATH,quiet=False,opath=OPATH,prefix='retina_',suffix=''):

    t0 = time.perf_counter()

    # Convert amira spatial graph files to json
    if convert_to_json:
        if quiet==False:
            print('')
        for f in ['retina_artery_lower.am','retina_vein_lower.am','retina_artery_upper.am','retina_vein_upper.am']:
            fname = join(data_path,f)
            if quiet==False:
                print('Converting AM to JSON: {}'.format(fname))
            convert(fname) #,opath=opath)
    
    # Run simulations
    if simulate:
        if quiet==False:
            print('')

        for f in ['upper','lower']:  
            fname = join(opath,'retina_artery_{}_discs_params.json'.format(f))
            if quiet==False:
                print('Simulating from {}'.format(fname))
            res = run_sim(fname,exe_path=exe_path,quiet=quiet)
            if res!=0:
                return False, fname
            
            # Convert csv output to amira graph format
            for v in ['vein','artery']:
                fname = join(path,f'{prefix}{v}_{f}_cco{suffix}.csv')
                if quiet==False:
                    print('Converting CSV to AM: {}'.format(fname))
                csv2amira(fname)

    if combine:
        # Combine graphs into one file
        mFiles = [  'retina_artery_upper_cco.csv.am',
                    'retina_vein_upper_cco.csv.am',
                    'retina_artery_lower_cco.csv.am',
                    'retina_vein_lower_cco.csv.am',
                 ]
        ofile = 'retina_cco.am'
        combine_graphs.combine_cco(opath,mFiles,ofile)

    t1 = time.perf_counter()
    if quiet==False:
        print(f"Simulation completed in {int(t1-t0)}s ({int((t1-t0)/60)} mins)")
        
    return True, ''
    
if __name__=='__main__':
    run_init()
