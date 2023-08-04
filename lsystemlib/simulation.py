# -*- coding: utf-8 -*-
"""
Created on Tue Dec 04 15:09:08 2018

@author: simon

"""

import numpy as np
#import cupy as cp
cp = np # Use numpy rather than cupy
import math
arr = np.asarray
import sys
import os
join = os.path.join
import re
import random
import json
import shutil
from matplotlib import pyplot as plt
from scipy import ndimage
import cProfile
import time
from stl import mesh # pip install numpy-stl
import nibabel as nib
import dill as pickle
import psutil
from multiprocessing import shared_memory, Process, Queue, Pipe, Pool

from retinasim import geometry
from retinasim.geometry import norm, cross
from retinasim.lsystemlib.utilities import *
from retinasim.lsystemlib.embedding import EmbedVessels
from retinasim.lsystemlib.tree import Tree
from retinasim.lsystemlib.growth_process import GrowthProcess, Params

num_cpus = psutil.cpu_count(logical=False)

sphere_centre = arr([0.,-20000.,0.])

np.random.seed()

class Simulation(object):

    def __init__(self,prefix='',path=None,store_graph=False,store_mesh=False,store_density=False,store_vegf=False,store_vessel_grid=False,store_every=1,
                      write_longitudinal=False,verbose=2,to_shared_memory=False,from_shared_memory=False,n=10000,logging=False,create_terminals=True,
                      domain_size=[1000.,1000.,1000.],domain=[None,None,None],domain_type='cuboid',ignore_domain=False,
                      embedding_resolution=None,embedding_dim=None,embed=True,embedding_offset=arr([0.,0.,0.]),tree=None,
                      initialise_folders=True,n_sampling_directions=21,planar=False,match_embedding_dims=False,max_cycles=50,
                      **kwargs):
                  
        self.verbose = verbose
        self.tree = tree
        self.set_domain(domain_size,domain=domain,domain_type=domain_type)
        self.current_cycle = 0
        
        self.params = Params()
                      
        self.max_cycles = max_cycles # Set to -1 for no limit
        
        self.store_graph = store_graph
        self.store_mesh = store_mesh
        self.store_vessel_grid = store_vessel_grid
        if path is None:
            initialise_folders = False
            self.path = None
        else:
            self.path = path
            if not os.path.exists(self.path):
                os.mkdir(self.path)
        
        self.prefix = prefix
        self.store_every = store_every
        self.store_vessel_grid_every = store_every*10.
        self.write_longitudinal = write_longitudinal
        
        # Directory structure for outputs
        self.initialise_directory_structure(initialise_folders=initialise_folders)
        
        # Generate growth sampling directions
        self.n_sampling_directions = n_sampling_directions # in theta and phi (odd numbers better?)
        self.planar = planar
        if self.planar:
            self.set_segment_sampling_directions(n=5,plane='xy')
            self.set_branching_sampling_directions(n=11,plane='xy')
        else:
            self.set_segment_sampling_directions(n=self.n_sampling_directions)
            self.set_branching_sampling_directions(n=self.n_sampling_directions)
                
        # Create tree object
        if self.tree is None:
            self.tree = Tree(verbose=verbose,domain=self.domain,ignore_domain=ignore_domain)

        # Create embedding object
        self.embed = embed
        if embedding_dim is not None and embedding_resolution is not None and self.embed:
            self.embed = True
            self.embed_vessels = EmbedVessels(ms=embedding_dim,resolution=embedding_resolution,offset=embedding_offset)
            if match_embedding_dims:
                self.set_domain(self.embed_vessels.domain_size,domain=self.embed_vessels.domain)
       
        # Some checks
        if self.params.get('current_segment_bias_angle')<np.rad2deg(self.segment_sampling_angular_displacement[0]):
            #print('Warning: Segment growth bias angle is less than sampling angular displacement. Growth bias may not have any influence!')
            pass
            
    def initialise_directory_structure(self, initialise_folders=True):
        if self.path is not None:
            self.gpath = os.path.join(self.path) #,'graph')
            self.mpath = os.path.join(self.path,'mesh')
            self.gridpath = os.path.join(self.path,'volume')
            self.midlinepath = os.path.join(self.path,'volume_midline')
        else:
            self.gpath = None
            self.mpath = None
            self.gridpath = None
            self.midlinepath = None       

        if initialise_folders:
            print('Initialising folders')
        
            if self.store_graph:
                self.clear_folder(self.gpath)
            elif initialise_folders:
                self.remove_folder(self.gpath)

            if self.store_mesh:
                self.clear_folder(self.mpath)
            elif initialise_folders:
                self.remove_folder(self.mpath)

            if self.store_vessel_grid:
                self.clear_folder(self.gridpath)
                self.clear_folder(self.midlinepath)
            elif initialise_folders:
                self.remove_folder(self.gridpath)
                self.remove_folder(self.midlinepath)

        
    def initialise_vessels(self):
        nvessel = 1
        for i in range(nvessel):    
            rndx0 = np.random.uniform(0.8,1.)
            rndx1 = np.random.uniform(0.8,1.)
            vessel_disp = 1000.
            
            # Select boundary face
            face = np.random.randint(0,5)
            face_normals = np.asarray([ [0.,0.,1.], [0.,0.,-1], [1.,0.,0.], [-1.,0.,0.], [0.,1.,0.], [0.,-1.,0.] ])
            x0,x1 = self.domain[0]
            xm = (x1-x0)/2.
            y0,y1 = self.domain[1]
            ym = (y1-y0)/2.
            z0,z1 = self.domain[2]
            zm = (z1-z0)/2.
            face_centres = np.asarray([ [xm,ym,z0], [xm,ym,z1], [x0,ym,zm], [x1,ym,zm], [xm,y0,zm], [xm,y1,zm] ])
            normal = face_normals[face]
            centre = face_centres[face]
            a0dir = normal.copy()
            off_axis_fraction = 0.2
            if a0dir[0]==0.:
                a0dir[0] += np.random.uniform(-off_axis_fraction,off_axis_fraction)
            if a0dir[1]==0.:
                a0dir[1] += np.random.uniform(-off_axis_fraction,off_axis_fraction)
            if a0dir[2]==0.:
                a0dir[2] += np.random.uniform(-off_axis_fraction,off_axis_fraction)

            a0loc = centre
            if normal[0]==0.:
                a0loc[0] = np.random.uniform(x0,x1)
            if normal[1]==0.:
                a0loc[1] = np.random.uniform(y0,y1)
            if normal[2]==0.:
                a0loc[2] = np.random.uniform(z0,z1)
                
            a0diameter = np.random.uniform(100.,500.)
            a0length = 500.
            b0_a = self.create_branch(diameter=a0diameter,length=a0length,direction=a0dir,start_coord=a0loc,category='artery') 
            
    def inside_domain(self,coord,segment=None,epsilon=1e-6,**kwargs):

        if self.domain is None or self.ignore_domain:
            return True
        # Cuboid domain
        if self.domain_type=='cuboid':
            if np.all(self.domain[:,0]<=coord) and np.all(self.domain[:,1]>=coord):
                return True
            else:
                #dom_int = self.find_domain_intersection(segment.start_node.coords,segment.end_node.coords)
                return False
        elif self.domain_type=='surface':
            # Assumes a plane oriented in x-y
            if np.all(self.domain[0:2,0]<=coord[0:2]) and np.all(self.domain[0:2,1]>=coord[0:2]):
                return True
            else:
                #dom_int = self.find_domain_intersection(segment.start_node.coords,segment.end_node.coords)
                return False
        elif self.domain_type=='sphere':
            radius = (self.domain[:,1]-self.domain[:,0])/2.
            centre = arr([0.,0.,0.])
            dist = np.linalg.norm(coord-centre)
            if dist>radius[0]:
                return False
            else:
                return True
        else:
            return True             
            
    def set_domain(self,domain_size,domain=[None,None,None],domain_type='cuboid',ignore_domain=False):
        self.domain_size = arr(domain_size) #um
        if domain[0] is None and domain_size[0] is not None:
            self.domain = arr([[-self.domain_size[0]/2,self.domain_size[0]/2],[-self.domain_size[1]/2,self.domain_size[1]/2],[0,self.domain_size[2]]]) # um
        else:
            self.domain = domain
        #if self.tree is not None:
        #    self.tree.domain = self.domain
        
        self.domain_type = domain_type
        self.ignore_domain = ignore_domain
            
    def set_branching_sampling_directions(self,n=51,plane=None):
        r = np.deg2rad([-180.,180.])
        
        if plane is None:
            theta = np.linspace(r[0],r[1],n)
            phi = np.linspace(r[0],r[1],n)
            directions = []
            sampling_angles = []
            for t in theta:
                for p in phi:
                    directions.append(geometry.rotate_3d(arr([0.,p,t],dtype='float'),zaxis))
                    sampling_angles.append([p,t])
            self.branching_sampling_directions = arr(directions)
            self.branching_sampling_angles = arr(sampling_angles)
            self.branching_sampling_angular_displacement = [np.abs(theta[1]-theta[0]),np.abs(phi[1]-phi[0])] 
        elif plane=='xy':
            theta = 0.
            phi = np.linspace(r[0],r[1],n)
            directions = []
            sampling_angles = []
            t = 0.
            for p in phi:
                directions.append(geometry.rotate_3d(arr([0.,p,t],dtype='float'),zaxis))
                sampling_angles.append([p,t])
            self.branching_sampling_directions = arr(directions)
            self.branching_sampling_angles = arr(sampling_angles)
            self.branching_sampling_angular_displacement = [0.,np.abs(phi[1]-phi[0])] 
        #print('Branching sampling angular resolution: {} deg'.format(np.rad2deg(self.branching_sampling_angular_displacement)))
        
    def set_segment_sampling_directions(self,n=51,plane=None):
        max_ang = self.params.get('max_segment_sampling_angle',None)
        if max_ang is None:
            max_ang = 10.
            
        r = np.deg2rad([-max_ang,max_ang])

        if plane is None:
            theta = cp.linspace(r[0],r[1],n)
            phi = cp.linspace(r[0],r[1],n)
            directions = []
            sampling_angles = []
            for t in theta:
                for p in phi:
                    directions.append(geometry.rotate_3d(arr([0.,p,t],dtype='float'),zaxis))
                    sampling_angles.append([p,t])
            self.segment_sampling_directions = arr(directions)
            self.segment_sampling_angles = arr(sampling_angles)
            self.segment_sampling_angular_displacement = [np.abs(theta[1]-theta[0]),np.abs(phi[1]-phi[0])]
        elif plane=='xy':
            rt = np.deg2rad([-1.,1.])
            theta = [0.] #cp.linspace(r[0],r[1],n)
            phi = cp.linspace(r[0],r[1],n**2)
            directions = []
            sampling_angles = []
            t = 0.
            for t in theta:
                for p in phi:
                    directions.append(geometry.rotate_3d(arr([0.,p,t],dtype='float'),zaxis))
                    sampling_angles.append([p,t])
            self.segment_sampling_directions = arr(directions)
            self.segment_sampling_angles = arr(sampling_angles)
            self.segment_sampling_angular_displacement = [0.,np.abs(phi[1]-phi[0])]
        #print('Segment sampling angular resolution: {} deg'.format(np.rad2deg(np.max(self.segment_sampling_angular_displacement[0]))))

    def get_segment_sampling_directions(self,current_direction):
        if np.all(current_direction==zaxis) or np.all(current_direction==-zaxis):
            return self.segment_sampling_directions
        # Rotate to current direction
        return self.rotate_sampling_directions_to_current(self.segment_sampling_directions,current_direction)
        
    def rotate_sampling_directions_to_current(self,sampling_directions,current_direction,centre_vector=None):
        if centre_vector is None:
            centre_vector = zaxis
        R = geometry.align_vectors(centre_vector,current_direction,homog=False)
        res = sampling_directions.copy()
        for i,s in enumerate(sampling_directions):
            res[i,:] = np.dot(R,s)
        return res
        
    def create_branch_callback(self,branch,**kwargs):
        return branch
        
    def create_branch(self,**kwargs):
        branch = self.tree.create_branch(**kwargs)
        
        if branch.tip_segment is not None:
            s0 = branch.tip_segment
            ins = self.inside_domain(s0.end_node.coords,segment=s0)
            if not ins:
                self.on_outside_domain(s0)
        
            if s0.status=='active':
                coll,coll_seg,intersection = self.tree.segment_collision_check(s0)
                if coll:
                    self.on_segment_collision(s0,coll_seg,intersection)
        
        branch = self.create_branch_callback(branch,**kwargs)
        return branch

    def create_vessel(self,length,origin=[0.,0.,0.],nn=3,category='artery',diameter=50.,connected=True,branch_status=2,direction=zaxis,start_cycle=0):
    
        start_coord = arr(origin)
        step_length = length // (nn-1)

        branch = self.tree.add_branch(type=type,order=order,category=category,cycle=self.current_cycle)
        
        points = []
        for i in range(nn):
            points.append(origin+direction*step_length*i)
        points = arr(points)
        end_coord = points[-1]

        segment = tree.add_segment_to_branch(branch,start_coord=start_coord,end_coord=end_coord,cycle=self.current_cycle)
        segment.edge.set_property('coords',points)
        
        return branch
        
# SIMULATION FUNCTIONS

    def update_edge_paths(self):
        pass

    def branch_transverse_to_world(self,branch_direction,angle_rad,sdirection,snormal):
        if not np.all(branch_direction==zaxis) and not np.all(branch_direction==-zaxis):
            rot_axis = cross(zaxis,branch_direction)
            angle = np.rad2deg(angle_rad)
            angle = np.random.normal(angle,self.params.branching_angle_sd)
            direction = geometry.rotate_about_axis(zaxis,rot_axis,angle)
            normal = geometry.rotate_about_axis(xaxis,rot_axis,angle)
            # Convert back to world reference frame
            direction /= norm(direction)
            sdirection /= norm(sdirection)
            snormal /= norm(snormal)
            world_direction = self.tree.branch_to_world(direction,sdirection,snormal)
            world_direction /= norm(world_direction)
            world_normal = self.tree.branch_to_world(normal,sdirection,snormal)
            world_normal /= norm(world_normal)
        else: # Branch axis already aligned with world z-axis
            #print('HERE!')
            direction = branch_direction
            world_direction = sdirection
            world_normal = snormal
            
        return world_direction, world_normal  
        
    def branch_to_world(self,branch_direction,branch_normal,sdirection,snormal):
        if not np.all(branch_direction==zaxis) and not np.all(branch_direction==-zaxis):
            # Convert back to world reference frame
            branch_direction /= norm(branch_direction)
            branch_normal /= norm(branch_normal)
            sdirection /= norm(sdirection)
            snormal /= norm(snormal)
            world_direction = self.tree.branch_to_world(branch_direction,sdirection,snormal)
            world_direction /= norm(world_direction)
            world_normal = self.tree.branch_to_world(branch_normal,sdirection,snormal)
            world_normal /= norm(world_normal)
        else: # Branch axis already aligned with world z-axis
            #print('HERE!')
            world_direction = sdirection
            world_normal = snormal
            
        return world_direction, world_normal
        
    def world_to_branch(self,world_direction,world_normal,sdirection,snormal):
        sdirection /= norm(sdirection)
        snormal /= norm(snormal)
        branch_direction = self.tree.world_to_branch(world_direction,sdirection,snormal)
        branch_direction /= norm(branch_direction)
        branch_normal = self.tree.world_to_branch(world_normal,sdirection,snormal)
        branch_normal /= norm(branch_normal)
            
        return branch_direction, branch_normal          
            
    def branch_initialiser(self,branch,first=False,segment=None,res_list=None,proc=None):
    
        # Do we need a branch? This function makes that decision so you don't have to.
    
        if proc is None:
            proc = branch.get_growth_process()
        
        if segment is None:
            segment = branch.tip_segment
      
        endpoints = None
        if self.artery_endpoints is not None:
            if branch.category=='artery':
                endpoints = self.artery_endpoints[self.artery_endpoints_occupied==0]
            elif branch.category=='vein':
                endpoints = self.vein_endpoints[self.vein_endpoints_occupied==0]
                
        if res_list is None:
            add_branch, res_list = proc.add_branch_here(branch,segment=segment,endpoints=endpoints,cycle=self.current_cycle)
        else:
            add_branch = True

        # If we need a branch, define its properties (branching angle, diameter, etc.)
        if add_branch:

            for res in res_list:
            
                # Mark endpoint as occupied
                if self.artery_endpoints is not None:
                    if branch.category=='artery':
                        endpoints = self.artery_endpoints
                    elif branch.category=='vein':
                        endpoints = self.vein_endpoints
                    mtch = [i for i,x in enumerate(endpoints) if res['end_point_coords'] is not None and np.all(res['end_point_coords']==x)]
                    if np.any(mtch):
                        if branch.category=='artery':
                            self.artery_endpoints_occupied[mtch] = 1
                        elif branch.category=='vein':
                            self.vein_endpoints_occupied[mtch] = 1
            
                if not res['fully_defined']: # If branching properties haven't been defined by the growth_process class
            
                    proc = res['creation_process']
            
                    last_branch_dist = branch.get_distance_from_last_branch(end_node=segment.end_node)
                
                    # See if the simulation provides a branch, based on world direction branching
                    wdir = self.calculate_world_branching_direction(branch, gen=res)
                    world_direction, world_normal = wdir[0], wdir[1] #['direction'],wdir['normal']
                    if len(wdir)==5:
                        angles = wdir[2]
                        world_direction_ongoing, world_normal_ongoing = wdir[3], wdir[4]
                        if world_direction_ongoing is not None:
            
                            branch.next_segment_params = {'diameter':None,'angle':-angles[0],
                                                          'world_direction':world_direction_ongoing,'world_normal':world_normal_ongoing,
                                                          'branch_direction':None,'branch_normal':None}
                                                          
                    if self.params.maximum_vessel_order_branching_behaviour=='stop':
                        if branch.order>=self.params.maximum_vessel_order:
                            return False, res                                                          
                        
                    if world_direction is None:
                        # Get transverse branching direction in branch coordinates (i.e. relative to growth direction)
                        gdir = branch.get_growth_direction()
                        branch_transverse_direction = self.calculate_transverse_branching_direction(branch)
                        
                        if self.params.bisecting_plane is None or branch.order<self.params.minimum_branching_order_for_bisecting_plane:
                            rot_axis = np.cross(zaxis,branch_transverse_direction)
                            # Get branching angle (angle to deviate from branching direction)
                            angles = proc.calculate_branching_angles(branch,gen=res)
                        else:
                            polarity = np.dot(branch_transverse_direction,xaxis)
                            if polarity>0.:
                                rot_axis_world = self.params.bisecting_plane[0]
                            elif polarity<=0.:
                                rot_axis_world = self.params.bisecting_plane[1]
                            rot_axis_world = rot_axis_world / norm(rot_axis_world)

                            rot_axis,_ = self.world_to_branch(rot_axis_world,xaxis,gdir['direction'],gdir['normal'])
                            rot_axis = rot_axis * polarity
                            rot_axis = rot_axis / norm(rot_axis)
                            if np.abs(np.dot(rot_axis,zaxis))==1.:
                                rot_axis = polarity*xaxis
                                
                            # Get branching angle (angle to deviate from branching direction)
                            angles = proc.calculate_branching_angles(branch,gen=res)                 
                        
                        # Rotate branch direction by second angle
                        branch_direction = geometry.rotate_about_axis(zaxis,rot_axis,angles[1])
                        # Define normals
                        branch_normal = branch.propogate_normal(gdir['direction'],branch_direction,gdir['normal']) #geometry.rotate_about_axis(xaxis,rot_axis,angles[1])

                        # Define ongoing direction for the parent branch
                        if angles[0] is not None:
                            branch_direction_ongoing = geometry.rotate_about_axis(zaxis,rot_axis,-angles[0])
                            branch_normal_ongoing = branch.propogate_normal(gdir['direction'],branch_direction_ongoing,gdir['normal']) #geometry.rotate_about_axis(xaxis,rot_axis,-angles[0]) # 
                        else:
                            branch_direction_ongoing, branch_normal_ongoing = gdir['direction'], gdir['normal']
                      
                        gdir = branch.get_growth_direction()
                        sdirection,snormal = gdir['direction'],gdir['normal']
                        
                        # Calculate world coordinates from branch coordinates
                        world_direction, world_normal = self.branch_to_world(branch_direction,branch_normal,sdirection,snormal)
                        # Calculate world coordinates from branch coordinates for the parent branch
                        world_direction_ongoing, world_normal_ongoing = self.branch_to_world(branch_direction_ongoing,branch_normal_ongoing,sdirection,snormal)
                        
                        if branch_transverse_direction is None:
                            branch_transverse_direction = arr([branch_direction[0],branch_direction[1],0.])
                        
                        branch.next_segment_params = {'diameter':None,'angle':-angles[0],
                                                      'world_direction':world_direction_ongoing,'world_normal':world_normal_ongoing,
                                                      'branch_direction':branch_direction_ongoing,'branch_normal':branch_normal_ongoing}
                    else:
                        gdir = branch.get_growth_direction()
                        branch_direction,branch_normal = self.world_to_branch(world_direction,world_normal,gdir['direction'],gdir['normal'])
                        branch_transverse_direction = arr([branch_direction[0],branch_direction[1],0.])

                    # Get vessel diameters
                    diameters = proc.calculate_branching_diameters(branch,direction=branch_direction,gen=res)
                    stub_branch = False

                    if branch.next_segment_params is not None:
                        branch.next_segment_params['diameter'] = diameters[0]
                        branch.next_segment_params['length'] = proc.calculate_segment_length(None,diameter=diameters[0])
                        if branch.next_segment_params['length'] is None:
                            branch.next_segment_params = None
                    #print('Next segment structure: {}'.format(branch.next_segment_params))

                    # Get length for next segment
                    length = proc.calculate_segment_length(branch,diameter=diameters[1])
                    if length is None or length<1e-6:
                        return False, res

                    res['world_direction'] = world_direction
                    res['world_normal'] = world_normal
                    res['branch_direction'] = branch_direction
                    res['branch_normal'] = branch_normal
                    res['branch_transverse_direction'] = branch_transverse_direction
                    res['diameters'] = diameters            
                    res['angles'] = angles            
                    res['length'] = length           
                    res['interbranch_distance'] = branch.interbranch_distance 
                    res['last_branch_distance'] = last_branch_dist
                    res['growth_process'] = branch.growth_process
                    res['branching_type'] = 't'
                    res['stub_branch'] = stub_branch
                    #return True, res
                
                    self.branch_initialiser_callback(branch,res)
                
                    if res['length']==0.:
                        breakpoint()
                
        return add_branch, res_list
        
    def initialise_branching_direction(self, branch):
        if self.params.branch_pattern_initial_direction is not None:
            direction = self.params.branch_pattern_initial_direction
        else:
            direction = np.array([np.random.uniform(-1.,1.), np.random.uniform(-1.,1.),0.])
        return direction / np.linalg.norm(direction)        
        
    def calculate_transverse_branching_direction(self,branch):

        prev_dir = branch.get_daughter_initialisation_branch_directions()
        epsilon = -1. + 1e-9
        
        polarity = 1
        
        if len(prev_dir)>1:
            dp = arr([np.dot(prev_dir[i-1],prev_dir[i]) for i,x in enumerate(prev_dir)])
        elif len(prev_dir)==1:
            dp = arr([-1.])
        else:
            direction = self.initialise_branching_direction(branch)
            return direction / np.linalg.norm(direction)   
        
        if self.params.branch_pattern=='alternate':

            if len(prev_dir)>1 and np.all(dp[-2:]<epsilon):
                # Alternate
                direction = prev_dir[-2]                
            elif len(prev_dir)>0 and prev_dir[0] is not None and np.all(dp[-1]<epsilon): # If only one previous, orthogonal branch
                #print(prev_dir[0])
                direction = geometry.rotate_about_axis(prev_dir[0],zaxis,180.)
            else: # No previous branches - define a random direction
                direction = self.initialise_branching_direction(branch)
                    
        elif self.params.branch_pattern=='spiral':

            if len(prev_dir)>3 and np.all(dp[-4:]<epsilon):
                direction = prev_dir[-4]
            elif len(prev_dir)>2 and np.all(dp[-3:]<epsilon):
                direction = prev_dir[-3]
            elif len(prev_dir)>2 and np.all(dp[-2:]<epsilon):
                if np.all(np.abs(dp[-1])<epsilon): # Re-initialise pattern
                    direction = geometry.rotate_about_axis(prev_dir[-1],zaxis,180.)
                else:
                    direction = np.cross(prev_dir[-2],prev_dir[-1])
            elif len(prev_dir)>0 and np.all(dp[-1:]<epsilon): # If only one previous branch
                direction = geometry.rotate_about_axis(prev_dir[-1],zaxis,180.)
            else: # No previous branches - define a random direction
                direction = self.initialise_branching_direction(branch)
                
        elif self.params.branch_pattern=='random':
            if len(prev_dir)>0:
                res = branch.get_daughter_directions_and_distances()
                if res is None:
                    import pdb
                    pdb.set_trace()
                angles = arr([np.arctan(x['direction'][1]/x['direction'][0]) for x in res])
                n = 180
                prob = np.zeros(n)
                branch_length = branch.get_length()
                xp = np.linspace(-np.pi/2,np.pi,n)
                for i,angle in enumerate(angles):
                    prob += (branch_length-res[i]['length'])*normprior(xp,mean=angle,sd=np.deg2rad(2.))
                prob = prob - np.min(prob)
                prob /= np.sum(prob)
                prob = 1. - prob
                prob /= np.sum(prob)
                angle = np.rad2deg(np.random.choice(xp,p=prob))
                direction = geometry.rotate_about_axis(xaxis,zaxis,angle)
                direction[0] *= np.random.choice([-1.,1.])
                direction[1] *= np.random.choice([-1.,1.])
                #import pdb
                #pdb.set_trace()
            else:
                direction = arr([np.random.uniform(-1.,1.), np.random.uniform(-1.,1.), 0.])
            
        elif self.params.branch_pattern=='terminal':
            self.temp_ind = -1
            terms = [i for i,x in enumerate(self.terminals) if self.terminal_occupied[i]==0]
            x0 = branch.tip_segment.end_node.coords            
            if len(terms)==0:
                return None
            dist = arr([norm(x-x0) for x in terms])
            prob = 1./dist
            prob = prob / np.sum(prob)
            ind = np.random.choice(terms,p=prob)
            terminal = self.terminals[ind]
            world_direction = terminal - x0
            world_direction = world_direction / norm(world_direction)
            gdir = branch.get_growth_direction()
            world_normal = branch.propogate_normal(gdir['direction'],world_direction,gdir['normal'])
            direction,normal = self.world_to_branch(world_direction,world_normal,gdir['direction'],gdir['normal'])
            direction = arr([direction[0], direction[1], 0.])
            self.temp_ind = ind
               
        direction = direction / np.linalg.norm(direction)                 
        #print('Trans dir: {}'.format(direction))
        
        return direction
        
    def calculate_world_branching_direction(self,branch,**kwargs):
        return None, None, None, None
    
    def get_angular_difference(self,current_direction,sampling_directions):
        current_direction_norm = current_direction / norm(current_direction)
        dp = np.clip(arr([np.dot(x,current_direction_norm)/norm(x) for x in sampling_directions]),-1,1)
        ang_dif_current = np.arccos(dp)
        return ang_dif_current                             
        
    def growth_bias(self,branch,current_direction,bias,range=90.,all_data=False,use_angle_prior=False,radial=False,resolve=True,bias_dir=None,bias_weights=[],sampling_directions=None,max_ang_dev=None):
    
        """
        Calculates biases in branching and growth directions
        All directions in *branch* coordinates with the third dimension (z) corresponding to the direction of branch growth
        """
    
        def _parse_nodes(end_coord,cur_cat):
            # Look for nearby nodes
            segments, points, npoints, cats, node_type, point_diameter, dist = self.tree.get_edge_points(distance_coord=end_coord)

            # Mutally attractive node criteria
            opp_cat_points = arr([i for i,x in enumerate(points) if (cats[i]!=cur_cat and node_type[i]!='inline')]) # Exclude branch and terminating nodes 
            dist_opp_cat = []
            if len(opp_cat_points)>0:
                dist_opp_cat = dist[opp_cat_points]
  
            # Node avoidance criteria
            same_cat_points = arr([i for i,x in enumerate(points) if (cats[i]==cur_cat or node_type[i]=='inline')]) 
            dist_same_cat = []
            if len(same_cat_points)>0:
                dist_same_cat = dist[same_cat_points]

            return segments,points,dist,node_type,cats,point_diameter,opp_cat_points,dist_opp_cat,same_cat_points,dist_same_cat

        code = ''
        
        end_node = branch.get_tip_node()
        end_coord = end_node.coords
        cur_cat = end_node.category
        
        if bias_dir is None:
            bias_dir = []
            bias_weights = []

        if sampling_directions is None:
            sampling_directions = self.branching_sampling_directions.copy() #.transpose()
        else:
            sampling_directions = sampling_directions.copy()
        direction_forbidden = np.zeros(sampling_directions.shape[0],dtype='bool')
        direction_forbidden[:] = False
        
        ang_dif_current = self.get_angular_difference(current_direction,sampling_directions)
        ang_prior_broad = normprior(ang_dif_current,sd=np.deg2rad(range))
        
        # Follow current direction using an angular probability distribution with a peak in the current direction
        if branch.terminal is not None:
            term_direction = branch.tip_segment.end_node.coords - branch.terminal
            term_direction /= norm(term_direction)
            term_dif_current = self.get_angular_difference(term_direction,sampling_directions)
            ang_prior = normprior(ang_dif_current,sd=np.deg2rad(self.params.get('current_segment_bias_angle')))
            A = branch.tip_segment.diameters[-1] # um
            ang_prior = ang_prior / A # normalise

            bias_dir.append(ang_prior)
            bias_weights.append(1.) #bias['terminal'])
            
        if 'upwards' in bias.keys() and bias['upwards']>0.:
            tip_direction = arr([0.,0.,0.02])
            ang_dif_temp = self.get_angular_difference(current_direction+tip_direction,sampling_directions)
            ang_prior = normprior(ang_dif_temp,sd=np.deg2rad(self.params.get('current_segment_bias_angle')))

            A = branch.tip_segment.diameters[-1] # um
            ang_prior = ang_prior / A # normalise

            bias_dir.append(ang_prior)
            bias_weights.append(bias['upwards'])
            
        if 'current' in bias.keys() and bias['current']>0.:

            ang_prior = normprior(ang_dif_current,sd=np.deg2rad(self.params.get('current_segment_bias_angle')))
            A = branch.tip_segment.diameters[-1] # um
            ang_prior = ang_prior / A # normalise

            bias_dir.append(ang_prior)
            bias_weights.append(bias['current'])
         
        opp_cat_points = None
        same_cat_points = None
        dist_opp_cat = None
        dist_same_cat = None
                    
        if 'same_vessel_avoidance' in bias.keys() and bias['same_vessel_avoidance']>0. and branch.tip_segment.diameters[-1]>=self.params.get('minimum_diameter_for_avoidance'):# and not bi in [0,1]: # and not np.all(self.node_density_rad[ni]==self.node_density_rad[ni][0]):  
            if opp_cat_points is None:
                segments,points,dist,node_type,cats,point_diameter,opp_cat_points,dist_opp_cat,same_cat_points,dist_same_cat = _parse_nodes(end_coord,cur_cat)                    
               
            # Avoid vessels of same kind    
            if len(dist_same_cat)>0:
                            
                import pdb
                pdb.set_trace()
                buffer = 5. # um
                proc = branch.get_generator()
                
                new_segment_length = proc.calculate_segment_length(branch)
                if new_segment_length is None:
                    import pdb
                    pdb.set_trace()
                look_ahead_distance = 200.*5
                sampling_directions_cp = sampling_directions.copy()
                # Get vessel diameters (from same vessel category)
                diameters = self.tree.node_diameter[same_cat_nodes]
                # Exclude nodes that are separated by more than a segment length length (plus buffering fraction)
                scn = same_cat_nodes[(dist_same_cat-diameters)<look_ahead_distance]
                # Exclude the source node
                scn = scn[scn!=ni]
                # Find segments containing the source node
                seg_ids_ni = self.tree.get_segment_containing_node(ni)
                ec_store = []
                # Source node coordinate and radius
                start_coord = np.asarray(self.tree.node_coords[ni],dtype='float')
                src_rad = self.tree.node_diameter[ni]/2.
                # Only process source branches that are active
                if len(scn)>0 and self.tree.branch_status[bi] in [1,2]: # self.tree.branch_type[bi]==1 and 
                
                    # Array for probability values (probability of growing in each direction)
                    prob = np.zeros(sampling_directions_cp.shape[0])
                    seg_store = []
                    seg_intersect = []
                    collision_with = np.zeros(sampling_directions.shape[0],dtype='int32') - 1
                    collision_details = []
                    deriv = np.zeros(sampling_directions_cp.shape[0])
                    
                    # Loop through each target node (filtered above)
                    for i,nj in enumerate(scn):
                        if nj!=ni: # This should always be true, as ni is removed from scn above, but will leave it here for now...
                        
                            # Get segments containing target node
                            seg_ids = self.tree.get_segment_containing_node(nj)
                            # Check it hasn't already been processed
                            seg_ids = [x for x in seg_ids if x not in seg_store]
                            radius = self.tree.node_diameter[nj] / 2
                            # Padding around vessel to include in vessel collision calculation
                            r_pad = self.params.vessel_avoidance_padding # um
                            #eff_rad = np.asarray(radius+src_rad+r_pad,dtype='float')
                            eff_rad = np.asarray(radius,dtype='float')
                            
                            # Loop through target segments
                            #print('Segment IDs: {}'.format(seg_ids))
                            for seg_id in seg_ids:
                                seg_store.append(seg_id)
                                if seg_id not in seg_ids_ni: # exclude any target segments that also contain the source node
                                    # Get target node radius and segment start/end coordinates

                                    cstart = np.asarray(self.tree.node_coords[self.tree.segment_start_node_id[seg_id]],dtype='float')
                                    cend = np.asarray(self.tree.node_coords[self.tree.segment_end_node_id[seg_id]],dtype='float')
                                    # Get target segment direction
                                    cdir = cend-cstart
                                    cdir = cdir/norm(cdir)
                                    # Loop through sampling directions
                                    for di,raydir in enumerate(sampling_directions_cp):
                                        # Detect direct collision between sampling ray projection and target segment
                                        # Calculate ray projection
                                        raydir /= norm(raydir)
                                        ec = np.asarray(start_coord + raydir*look_ahead_distance,dtype='float')
                                        #if False:
                                        #    ec,ray_dir = project_vector_onto_sphere(start_coord,ec,sphere_centre)
                                        ec_store.append(ec)
  
                                        # Collision calculation
                                        #intersection,intersection_type,inside = geometry.segment_cylinder_collision(cstart,cend,eff_rad,start_coord,ec)
                                        _,_,intersection = geometry.cylinder_collision(cstart,cend,eff_rad,start_coord,ec,src_rad,n=4)
 
                                        if len(intersection)>0:# or inside[1]: # or np.any(inside):
                                            intersection = [x for x in intersection if norm(x-end_coord)>2.*src_rad]
                                                  
                                            if len(intersection)>0:            
                                                #print('Direction forbidden!')
                                                direction_forbidden[di] = True
                                                for intr in intersection:
                                                    seg_intersect.append(intr)
                                                collision_with[di] = seg_id
                                           
                                        if not direction_forbidden[di]:
                                            if False:
                                                prob[di] = 1
                                            else:
                                                # Ray overlap with segment normal
                                                seg_com = np.mean([cstart,cend],axis=0)
                                                com_ray = seg_com-start_coord
                                                dp = np.dot(raydir,com_ray/norm(com_ray))
                                                if True: #dp>=0.:
                                                    # Weight by distance (closer is lower prob)
                                                    A = radius * 2. # um
                                                    b = 1./200. # /um

                                                    f_src = A * np.exp(-b * norm(end_coord-seg_com))
                                                    f_tar = A * np.exp(-b * norm(ec-seg_com))
                                                    prob[di] += norm(ec-end_coord) / (f_tar - f_src) # 1/deriv
                                                    
                                                    #cost = (A * np.exp(-b * norm(ec-seg_com)) - A * np.exp(-b * norm(end_coord-seg_com))) / norm(ec-end_coord)
                                                    #prob[di] += 1./cost  #np.square(norm(com_ray))/dp
                                                    #print('Derivative weighting, sample {}, seg {}, dist {}: {}, f_src {}, f_tar {}'.format(di,seg_id,norm(ec-seg_com),prob[di],f_src,f_tar))
                                            
                    
                    if np.all(direction_forbidden):
                        print('All directions forbidden')
                        return None,None,None,None,'forbidden'   
                    
                    def _plot_prob(bias_dir, scale_by_prob=True):
                        #cols = ['blue','green','red']
                        ax = None
                        # Plot all segments in collision detection
                        for seg in seg_store:
                            x0,x1 = self.tree.node_coords[self.tree.segment_start_node_id[seg]], self.tree.node_coords[self.tree.segment_end_node_id[seg]]
                            if seg in seg_ids_ni: # Blue for source segment
                                col = 'blue'
                            elif seg in collision_with:
                                col = 'yellow'
                            else:
                                col = 'red' # Red for target segments
                            R = self.tree.node_diameter[self.tree.segment_start_node_id[seg]]
                            ax = plot_cylinder(x0,x1,R,ax=ax,c=col)
                            ax = plot_3d_vector(x0,x1,c=col,ax=ax)
                        
                        # Plot intersection points
                        for x in seg_intersect:
                            try:
                                x = x.squeeze()
                                ax.scatter(x[0],x[1],x[2],c='yellow')
                            except Exception as e:
                                print(e)
                                import pdb
                                pdb.set_trace()

                        # Plot end coordinates of rays
                        for j,ec in enumerate(ec_store):
                            ax.scatter(ec[0],ec[1],ec[2],c='b')
                        
                        for bii,bd in enumerate(bias_dir):
                            prb = self.bias_to_prob(bd,uni_if_null=True)
                            #print('Prob {}: {}'.format(bii,prb))
                            
                            for j in np.arange(sampling_directions_cp.shape[0]): 
                                curdir = sampling_directions_cp[j]
                                x0 = end_coord


                                if scale_by_prob:
                                    x1 = start_coord + curdir*prb[j]*buffer*new_segment_length 
                                else:                                   
                                    x1 = start_coord + curdir*new_segment_length
                                if direction_forbidden[j]:
                                    col = 'yellow'
                                else:
                                    col = 'green'
                                ax = plot_3d_vector(x0,x1,ax=ax,c=None)
                                #ax.scatter(dirs[j,0]*1,dirs[j,1]*1,dirs[j,2]*1)
                                
                        # Plot source segment start / end
                        ax.scatter(start_coord[0],start_coord[1],start_coord[2],c='yellow')
                        set_axes_equal(ax)
                                
                        plt.show()
                    
                    prob = np.max(prob) - prob
                    prob[direction_forbidden] = 0.
                    #if use_angle_prior:
                    #    prob += ang_prior_broad
                    if np.sum(prob)>0: # Only add in bias if sum is > 0
                        prob = prob / np.sum(prob)
                        bias_dir.append(prob)
                        bias_weights.append(bias['same_vessel_avoidance'])
                    
                    plot_prob = False
                    #if self.current_cycle>20:
                    #    plot_prob = True
                    if plot_prob: 
                        _plot_prob(bias_dir)

        # Apply weighting to bias and sum
        if np.sum(bias_weights)>0.:              
            bias_weights = bias_weights / np.sum(bias_weights) # Normalise weights
            bias_dir = np.product([bd*bias_weights[i] for i,bd in enumerate(bias_dir) if bias_weights[i]>0.],axis=0)
        else:
            return current_direction,bias_weights,sampling_directions,None,'bias_weighting_error'

        if np.all(direction_forbidden):
            return None,None,None,None,'forbidden'
            
        bias_dir = bias_dir[direction_forbidden==False]
        sampling_directions = sampling_directions[direction_forbidden==False]

        if resolve:
            bias_dir,code = self.resolve_sampling_bias(bias_dir,sampling_directions,current_direction,max_ang_dev=max_ang_dev)
  
        return bias_dir,bias_weights,sampling_directions,None,code
        
    def bias_to_prob(self,bias_dir,uni_if_null=False):
        def uni_prob_res(bias_dir):
            print('Warning, default flat probability dist!')
            return np.ones(len(bias_dir)) / float(len(bias_dir))
            
        if np.nansum(bias_dir)==0:
            if uni_if_null:
                return uni_prob_res(bias_dir)
            return None            
        prob = bias_dir / np.nansum(bias_dir)
        prob[np.isnan(prob)] = 0.
        if np.sum(prob)<=0.:
            if uni_if_null:
                return uni_prob_res(bias_dir)                   
            return None
        prob = prob / np.sum(prob)
        if np.any(np.isnan(prob)):                    
            if uni_if_null:
                return uni_prob_res(bias_dir)
            return None
        else:
            return prob        
        
    def resolve_sampling_bias(self,bias_dir,sampling_directions,current_direction,max_ang_dev=None):
               
        mode = self.params.sampling_mode #'probabilistic' # Needs to be probabilistic!
        if mode=='probabilistic':
            prob = self.bias_to_prob(bias_dir,uni_if_null=True)
            if prob is None:                    
                return None,'prob error'
            bias_dir = sampling_directions[np.random.choice(len(sampling_directions),p=prob)]
        elif mode=='maximum':
            bias_dir = sampling_directions[np.argmax(bias_dir)]
        # Normalise bias
        bias_dir = bias_dir / norm(bias_dir)
        
        current_direction = current_direction / norm(current_direction)
        
        # Calculate angle between current vessel direction and sampling direction
        ang = np.arccos(np.clip(np.dot(bias_dir,current_direction),-1,1))

        if max_ang_dev is None:
            max_ang_dev = np.max(self.sampling_angular_displacement)*2
        if max_ang_dev and ang>max_ang_dev:
            # Rotate to closest direction with the maximum allowed angular deviation
            ang_dif_current = np.abs(self.get_angular_difference(current_direction,sampling_directions))
            valid_directions = sampling_directions[ang_dif_current<max_ang_dev]
            if len(valid_directions)==0:
                return None,'forbidden'
            bias_dir = valid_directions[np.argmax(np.dot(bias_dir,valid_directions.transpose())),:]
            
        return bias_dir,''
        
    def process_segment(self,segment,cycle=0,add_branch=None,res_list=[]):
    
        change_in_cycle = False
        branch = segment.branch
        
        if add_branch is None:
            add_branch,res_list = self.branch_initialiser(branch,segment=segment)
        
        # If so, create the branch
        new_branch = None
        if add_branch:
            for res in res_list:                
                if res['length']<1e-6:
                    breakpoint()
        
                # Attempt to calculate branch end node coordinate, if growth process allows it
                if res['end_point_coords'] is None:
                    end_coord = self.calculate_branch_end_node_coords(segment.end_node.coords,res['world_direction'],res['length'])
                    res['end_point_coords'] = end_coord
                    
                if res['node'] is None:
                    res['node'] = segment.end_node
                
                #print('End coord:{}, diameter:{}'.format(end_coord,res['diameters']))
                                
                new_branch = self.create_branch(     parent_branch=branch,
                                                     direction=res['world_direction'],
                                                     normal=res['world_normal'],
                                                     length=res['length'],
                                                     diameter=res['diameters'][1],
                                                     initial_direction_brf=res['branch_transverse_direction'],
                                                     start_node=res['node'],
                                                     label=res['label'],
                                                     cycle=self.current_cycle,
                                                     stub=res['stub_branch'],
                                                     creation_process=res['creation_process'],
                                                     end_coord=res['end_point_coords'],
                                                     gen=res    
                                                     )

                if new_branch is not None:
                    #print('ADD BRANCH, id:{}, order:{}, ibd:{}, lbd:{}, length:{}'.format(new_branch.id,new_branch.order,res['interbranch_distance'],res['last_branch_distance'],res['length']))
                    change_in_cycle = True
                    
        return change_in_cycle, new_branch
        
    def process_branch(self,branch,cycle=0):

        change_in_cycle = False
            
        if cycle<branch.start_cycle:
            return  
            
        branch_nodes = branch.get_nodes()                

        # Add a tip branch
        if self.params.enable_tip_branching and branch.status=='active': # If actively growing
            # See if a new branch is required
            add_branch,res_list = self.branch_initialiser(branch)
            
            # If so, create the branch
            if add_branch:
                for res in res_list:                
                    if res['length']<1e-6:
                        breakpoint()
            
                    # Attempt to calculate branch end node coordinate, if growth process allows it
                    if res['end_point_coords'] is None:
                        end_coord = self.calculate_branch_end_node_coords(branch.tip_segment.end_node.coords,res['world_direction'],res['length'])
                        res['end_point_coords'] = end_coord
                        
                    if res['node'] is None:
                        res['node'] = branch.tip_segment.end_node
                    
                    #print('End coord:{}, diameter:{}'.format(end_coord,res['diameters']))
                                    
                    new_branch = self.create_branch(     parent_branch=branch,
                                                         direction=res['world_direction'],
                                                         normal=res['world_normal'],
                                                         length=res['length'],
                                                         diameter=res['diameters'][1],
                                                         initial_direction_brf=res['branch_transverse_direction'],
                                                         start_node=res['node'],
                                                         label=res['label'],
                                                         cycle=self.current_cycle,
                                                         stub=res['stub_branch'],
                                                         creation_process=res['creation_process'],
                                                         end_coord=res['end_point_coords'],
                                                         gen=res    
                                                         )

                    if new_branch is not None:
                        #print('ADD BRANCH, id:{}, order:{}, ibd:{}, lbd:{}, length:{}'.format(new_branch.id,new_branch.order,res['interbranch_distance'],res['last_branch_distance'],res['length']))
                        change_in_cycle = True

        # APICAL GROWTH at branch tip
        tip_node = branch.get_tip_node()
        # Define the interbranch distance for the current branch (if not possible, return)
        proc = branch.get_growth_process()
        proc.set_interbranch_distance(branch)
        if tip_node is not None and branch.status=='active' and branch.interbranch_distance is not None:                
            # Get the tip segment length since last branch (or root node)
            current_segment_length = branch.tip_segment.get_length()
            last_branch_distance = branch.get_distance_from_last_branch()

            segment,code = self.add_segment_to_branch(branch,ignore_domain=True)
            if segment is not None:
                if code in ['domain_termination','outside domain','minimum']:
                    #print('***INACTIVE 1')
                    branch.status = 'inactive'
                    change_in_cycle = True                
                elif code=='domain_error':
                    branch.status = 'inactive'
                elif code!='':
                    #print('***INACTIVE 2')
                    branch.status = 'inactive'
                else:
                    change_in_cycle = True
                if code!='' and self.verbose>1:
                    print('BRANCH ADD SEG FAIL CODE: {}'.format(code))
                    
            else:
                #import pdb
                #pdb.set_trace()
                pass
        else:
            pass
            #if tip_node is None:
            #    print('Error: tip segment missing!')
                
        # Create edge paths
        if False:
            self.update_edge_paths()
                    
        #self.tree.consistency_check()
        return change_in_cycle  
        
    def plot_sample_distribution(self,dirs,prob):
    
        cols = ['blue','green','red']
        ax = None
        for j in np.arange(dirs.shape[0]): 
            pp = prob[j] / np.max(prob)
            if pp>0.01:
                #for i in np.arange(sdtr.shape[0]): 
                v = dirs[j,:]*pp
                ax = plot_3d_vector([0.,0.,0],v,ax=ax)
            #ax.scatter(dirs[j,0]*1,dirs[j,1]*1,dirs[j,2]*1)
            #ax = plot_3d_vector([0.,0.,0],sdirection,ax=ax,c='black')
        #ax.scatter([-1,-1,-1],[1,1,1],c='black')
        #plt.xlim((-1,1))
        #plt.ylim((-1,1))
        mn,mx = -1,1
        ax.set_xlim((mn,mx))
        ax.set_ylim((mn,mx))
        ax.set_zlim((mn,mx))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        
    def calculate_branch_end_node_coords(self,start_coord,branch_direction,length):
        return None # for subclass use (e.g. snapping to a surface)
        
    def add_branch(self, parent, direction=arr([0.,0.,1.]), diameter=1e4, length=2., label=''):
    
        normal = parent.propogate_normal(parent.growth_direction,direction,parent.growth_normal)            
        
        if label=='':
            label = parent.label
            
        branch_direction, branch_normal = self.world_to_branch(direction,normal,parent.growth_direction,parent.growth_normal)

        return self.create_branch(    parent_branch=parent,
                                      direction=direction,
                                      normal=normal,
                                      length=length,
                                      diameter=diameter,
                                      #end_coord=end_coord, # might cause bugs - needs testing!
                                      initial_direction_brf=arr([branch_direction[0],branch_direction[1],0.]),
                                      start_node=parent.tip_segment.end_node,
                                      label=label,
                                      cycle=self.current_cycle)
                                      
    def add_branch_to_segment(self, seg, growth_process=None, fr=0.5):                          

        #pts = seg.get_edge_coords()
        #if pts.shape[0]==2:
        #    vec = seg.end_node.coords - seg.start_node.coords
        #    vec /= np.linalg.norm(vec)
        #    div_coord = seg.start_node.coords + fr*vec    
        
        #self.branch_initialiser(seg.branch,first=False,segment=seg,res_list=growth_process) 
        return self.process_segment(seg,add_branch=True,res_list=growth_process)
        
        
    def add_segment_to_branch_callback(self,branch,**kwargs):
        return branch
        
    def branch_initialiser_callback(self, branch, res):
        return res
        
    def on_segment_collision(self, segment, coll_seg, intersection,**kwargs):
    
        # Decide what to do in the event of collisions
        
        branch1 = segment.branch
        d1 = segment.diameters[-1]
        d2 = coll_seg.diameters[-1]
        
        if coll_seg.branch.category==segment.branch.category:

            if d1<d2:
                branch1.set_property('status','inactive')
                branch1.remove_segment(segment)
                segment = None
                
            branch1.set_property('status','inactive')
            #branch2.set_property('status','inactive')

        else: # Disallow vein-artery collisions if both vessels are small
            coll_diameter = 30.
            if d1<coll_diameter and d2<coll_diameter:
                branch1.set_property('status','inactive')
                branch1.remove_segment(segment)
                segment = None
                
            #branch1.set_property('status','inactive')
            #branch2.set_property('status','inactive')
                
    def on_outside_domain(self, segment,intersection=None,**kwargs):

        if intersection is None:
            dir = segment.end_node.coords - segment.start_node.coords
            dir /= norm(segment.end_node.coords - segment.start_node.coords)
            intersection = self.find_domain_intersection(segment.start_node.coords,segment.end_node.coords)
         
        intersection = None
        if intersection is None:
            #breakpoint()
            #segment.branch.set_property('status','inactive')
            self.tree.remove_segment(segment)
            return
        
        segment.end_node.set_property('coords',intersection)
        if segment.branch.tip_segment==segment:
            segment.branch.set_property('status','inactive')
              
    def add_segment_to_branch(self,branch,length=None,diameter=None,start_coord=None,end_coord=None,
                                   direction=None,normal=None,category=None,generator=None,label=None,ignore_domain=False):

        """
        Adds a segment to an existing branch (e.g. during apical growth)
        Also maintains list of branch initial directions, for use during the growth of other branches (e.g. for maintaining an alternating branching pattern)
        Input directions are in *world* coordinates
        """
        
        #print(len(branch.segments),len(branch.daughters))
        if len(branch.segments)>1 and len(branch.daughters)==0:
            branch.status = 'inactive'

        if branch.status!='active':
            return None, 'branch not active'

        code = ''
        gdir = branch.get_growth_direction()
        
        tip_node = branch.tip_segment.end_node

        # Normal bias is to maintain current branch direction
        
        if branch.next_segment_params is not None:
            #print('Next seg diameter: {}'.format(
            diameter = branch.next_segment_params['diameter']
            current_direction, current_normal = branch.next_segment_params['world_direction'], branch.next_segment_params['world_normal']
            branch.next_segment_params = None
        elif direction is not None:
            world_direction = direction
            if normal is None:
                world_normal = branch.propogate_normal(gdir['direction'],world_direction,gdir['normal']) 
            else:
                world_normal = normal
            current_direction, current_normal = world_direction, world_normal
        else:
            current_direction,current_normal = gdir['direction'],gdir['normal']
   
        if self.params.segment_bias is not None and gdir['direction'] is not None:
            # Get correct bias
            bias = self.params.segment_bias
            sampling_directions = self.get_segment_sampling_directions(current_direction)
            max_dev = self.params.max_segment_angle_per_step
            bias_dir,bias_weights,sample_directions,s0,code = self.growth_bias(branch,current_direction,bias,sampling_directions=sampling_directions,max_ang_dev=max_dev)
            if code=='connected':
                return s0,code
            elif bias_dir is None and s0 is None:
                return None,code
        else:
            # With no bias specified, maintain growth in the current direction
            bias_dir = current_direction / norm(current_direction)

        # Find new direction, based on biases
        direction_sd = self.params.apical_growth_direction_sd
        if direction_sd>0.:
            new_dir = bias_dir #+ np.random.normal(0.,direction_sd,3)
        else:
            new_dir = bias_dir
            
        # Convert bias direction to branch coordinates
        new_dir = new_dir / norm(new_dir)
        if np.any(np.isnan(new_dir)):
            breakpoint()

        if generator is None: # usually the case
            if diameter is None:
                if branch.next_diameter is not None:
                    diameter = branch.next_diameter
                    self.tree.branch_next_diameter[branch_id] = -1.
                elif self.params.inherit_diameter_segment is not None:
                    diameter = branch.tip_segment.diameters[-1]
                else:
                    diameter = self.tree.unit_diameter

            #if self.params.stop_growth_at_minimum_vessel_size and np.any(diameter<self.params.minimum_vessel_diameter):
            #    return None, 'minimum diameter'
            diameters = branch.tip_segment.diameters
            if self.params.minimum_vessel_diameter_growth_behaviour=='stop':
                if diameters[-1]<self.params.minimum_vessel_diameter:
                    return False, 'minimum'
            elif self.params.minimum_vessel_diameter_growth_behaviour=='clip':
                if diameters[-1]<self.params.minimum_vessel_diameter:
                    diameter = np.clip(diameter,self.params.minimum_vessel_diameter,None) 
                    
            if self.params.diameter_rand_frac is not None:
                diameter += np.random.normal(0.,diameter*self.params.diameter_rand_frac)

            if category is None:
                category = branch.category

            length_was_none = False
            if length is None:
                length_was_none = True
                proc = branch.get_growth_process()
                length = proc.calculate_segment_length(branch,diameter=diameter)
                if length is None:
                    return None, 'invalid segment length'
             
            # See if a bespoke end coordinate fucntion has been defined. If not, None is returned and end_coord is calculated by tree class
            end_coord = self.calculate_branch_end_node_coords(branch.tip_segment.end_node.coords,new_dir,length)
            
            s0, code = self.tree.add_segment_to_branch(branch,direction=new_dir,length=length,end_coord=end_coord,diameter=diameter,cycle=self.current_cycle,ignore_domain=True) 
            ins = self.inside_domain(s0.end_node.coords,segment=s0)
            if not ins:
                self.on_outside_domain(s0)
            
            if s0.status=='active':
                coll,coll_seg,intersection = self.tree.segment_collision_check(s0)
                if coll:
                    self.on_segment_collision(s0,coll_seg,intersection)
        else:
            s0,code = generator(branch, new_dir)
            
        branch = self.add_segment_to_branch_callback(branch,s0)

        return s0,code   
        
    def find_domain_intersection(self,start_coord,end_coord,epsilon=1e-6):
        # Find which face segment intersects
        if self.domain_type=='cuboid':
            # Point on each face
            # Normal to each face
            pn = arr([[0.,0.,1.],[0.,-1.,0],[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.],[-1.,0.,0.]])
            intersection = np.zeros([6,3]) * np.nan
            for i in range(6): 
                pnt = arr([(x[1]-x[0])/2. for x in self.domain])
                ind = np.where(pn[i]==1)
                if len(ind[0])>0:                
                    pnt[ind[0]] = self.domain[ind[0],0]
                ind = np.where(pn[i]==-1)
                if len(ind[0])>0:                
                    pnt[ind[0]] = self.domain[ind[0],1]
                    
                dir = end_coord - start_coord
                coll,pt = geometry.line_plane_intersection(pnt,pn[i],end_coord,dir)
                # See if intersection point is inside the domain
                if pt is not None and self.inside_domain(pt):
                    intersection[i,:] = pt
            # Find closest intersection that is on the surface of the cubic domain
            if np.all(~np.isfinite(intersection)):
                return None
            face = np.nanargmin([norm(x-end_coord) for x in intersection])
            coord = intersection[face,:]
            return coord  
        else:  
            return None       

    def run(self, start_cycle=1,parallel=True,resume=False,consistency_check=True,segment_update=True,min_segment_update_cycle=8,branch_update=True,write=True):

        meshcount = 0
                
        sim_t0 = time.time()
        self.inline_branches = []
        
        if start_cycle is None or resume: # Continue
            start_cycle = self.current_cycle + 1
    
        tpb,tc = None,None # time per branch, time per cycle
        for cycle in range(start_cycle,start_cycle+self.max_cycles):

            t0 = time.time()
            self.current_cycle = cycle
            
            branches = self.tree.branches
            active_branches = self.tree.get_active_branches()
            segs = self.tree.segments
            nsegs_start = len(segs)
            nodes = self.tree.get_node_coords(active=True)                    
            n_branches_removed = 0
            
            change_in_cycle = False
            
            if False: #self.verbose>0:
                print('Starting cycle {0}, {1} branches ({2} active), {3} segments, tpb:{4}, tc:{5}'.format(cycle,len(branches),len(active_branches),len(segs),tpb,tc))

            for branch in active_branches:
                tb = time.time()
                br_change_in_cycle = self.process_branch(branch,cycle=cycle) #.remote
                if br_change_in_cycle:
                    change_in_cycle = True
                    
                if consistency_check:
                    cc_res,msg = self.tree.consistency_check()
                    if not cc_res:
                        breakpoint()
                        pass
                        
                    #colls = self.tree.collision_check(same_kind=True)
                    #if len(colls)>0:
                    #    for coll in colls:
                    #        print('Segment collision: branch1, seg1: {},{}, branch2, seg2: {},{}'.format(coll[0].branch.id,coll[0].id,coll[1].branch.id,coll[1].id))
                    #    breakpoint()                    
                    
            if consistency_check:
                cc_res,msg = self.tree.consistency_check()
                if not cc_res:
                    #import pdb; pdb.set_trace()
                    pass

            if len(active_branches)>0:
                tpb = (time.time()-t0)/len(active_branches)
            else:
                tpb = 0.
            tc = time.time()-t0
            
            if segment_update and cycle>min_segment_update_cycle:
                segs = [x for x in self.tree.segments]
                for seg in segs:
                    if seg!=seg.branch.tip_segment:
                        seg_change_in_cycle,br = self.process_segment(seg,cycle=cycle) #.remote
                        if seg_change_in_cycle:
                            change_in_cycle = True

            if self.write_longitudinal:
                self.write()

            if len(self.tree.segments)==nsegs_start:
                change_in_cycle = False
            
            dt = time.time() - t0
            nbranch = len(branches)
            if nbranch>0:
                tpb = dt / float(nbranch)
            else:
                tpb = 0.
            
            # Create edge paths
            if False:
                self.update_edge_paths()
                import pdb
                pdb.set_trace()

            if change_in_cycle==False and self.exit_on_no_change:
                #print('No change in cycle {}, exiting...'.format(cycle))
                break
            elif self.max_cycles>=0 and cycle>=self.max_cycles:
                #print('Maximum number of cycles reached ({}), exiting...'.format(cycle))
                break
            else:
                if self.embed:
                    new_segments = np.where(self.tree.segment.cycle==self.current_cycle)
                    for i in new_segments[0]:
                        self.embed_vessels.embed_vessel_in_grid(self.tree,self.tree.segment_id[i])
                        
                    
            test = self.tree.consistency_check()
            #print('Consistency test result: {}'.format(test))
            if not test[0]:
                #import pdb; pdb.set_trace()
                pass
                   
        # END CYCLE LOOP
        
        # Create edge paths
        if False:
            self.update_edge_paths()
        
        if parallel:
            pqueue.put('DONE')
            reader_p.join()         # Wait for the reader to finish
            
        sim_time = time.time() - sim_t0
        #print('Simulation run time: {}s'.format(round(sim_time,2)))
        #print('Branch density: max={}'.format(np.max(self.branch_density)))
        
        if write:
            self.write(final=True)
        
    def create_filename(self,cycle,name,extension,prefix='',suffix='',longitudinal=True):
        if longitudinal and cycle>=0:
           cystr = '_cycle'+str(cycle).zfill(5)
        else:
           cystr = ''

        return '{}{}{}{}.{}'.format(prefix,name,cystr,suffix,extension)
    
    def write_pickle(self, filepath):
    
        with open(filepath,'wb') as fh:
            pickle.dump(self, fh)
        
    def write(self, final=False):

        if True: #self.store_graph:
            if True: #self.current_cycle%self.store_every==self.store_every-1:
                filename = self.create_filename(self.current_cycle,'','am',prefix=self.prefix,longitudinal=self.write_longitudinal)
                gfile = join(self.gpath,filename)
                self.tree.write_amira(gfile)
                print('Written: {}'.format(gfile))
                
        if self.store_vessel_grid and self.embed and final:
            if True:
            #if self.current_cycle%self.store_vessel_grid_every==self.store_vessel_grid_every-1:
                filename = self.create_filename(self.current_cycle,'','nii',prefix=self.prefix,longitudinal=self.write_longitudinal)
                dfile = join(self.gridpath,filename)
                self.embed_vessels.write_vessel_grid(dfile)
                
                filename = self.create_filename(self.current_cycle,'','nii',prefix=self.prefix,suffix='_midline',longitudinal=self.write_longitudinal)
                dfile = join(self.midlinepath,filename)
                self.embed_vessels.write_midline_grid(dfile)

        # Write mesh files
        if self.store_mesh: #self.store_vessel_grid:
            branches = self.tree.get_branches()
            bcount = 0
            branch_mesh = None
            for bi in branches:
                bm = self.tree.tube_mesh(self.tree.node_coords[self.tree.branch_node_ids[bi]], self.tree.node_diameter[self.tree.branch_node_ids[bi]]/2.)
                if bcount==0:
                    branch_mesh = bm
                else:
                    branch_mesh = mesh.Mesh(np.concatenate([branch_mesh.data, bm.data]))
                bcount += 1
                             
            meshtype = 'stl'
            if meshtype=='stl':
                filename = self.create_filename(self.current_cycle,'','stl',prefix=self.prefix,longitudinal=self.write_longitudinal)  
                mfile = join(self.mpath,filename)
                #print('Writing branch-only mesh to {}'.format(mfile))
                branch_mesh.save(mfile)
            elif meshtype=='ply':
                import pdb
                pdb.set_trace()
                vertex = np.array(branch_mesh.vectors)
                
        # write pickle
        #if True:
        #    pickle_filename = self.create_filename(self.current_cycle,'','p',prefix=self.prefix,longitudinal=self.write_longitudinal)  
        #    gfile = join(self.gpath,pickle_filename)
        #    self.write_pickle(pickle_filename)
        #    print('Pickle written to {}'.format(pickle_filename))
        
    def clear_folder(self,folder):
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                try:
                    os.remove(os.path.join(folder,filename))
                except Exception as e:
                    print('Could not delete {}. Error: {}'.format(filename,e))
        else:
            os.mkdir(folder)
            
    def remove_folder(self,folder):
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                try:
                    os.remove(os.path.join(folder,filename))
                except Exception as e:
                    print('Could not delete {}. Error: {}'.format(filename,e))
            os.rmdir(folder)
    
  
