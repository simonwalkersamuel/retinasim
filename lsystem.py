"""
Creates an L-system seed for the CCO algorithm
"""

import numpy as np
arr = np.asarray
norm, uni = np.random.normal, np.random.uniform
from pymira import spatialgraph #, plot_graph
#from mayavi import mlab
import matplotlib as mpl
from opensimplex import OpenSimplex
import uuid
from scipy.stats import multivariate_normal
from functools import partial
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
join = os.path.join
from scipy.interpolate import griddata
from matplotlib import cm
import open3d as o3d
import dill  

from retinasim.lsystemlib.simulation import Simulation, GrowthProcess
from retinasim.lsystemlib import simulation
from retinasim import geometry
from retinasim.geometry import norm, cross
from retinasim.lsystemlib.embedding import EmbedVessels
from retinasim.lsystemlib.utilities import *
from retinasim import fovea_sim, retina_surface
from retinasim.eye import Eye
from retinasim.lsystemlib.endpoint_seed_generator import boid_seed_generator

# Default values
ext = 35000./2.
xoffset = 0
DOMAIN = arr([[-ext+xoffset,ext+xoffset],[-ext,ext],[-ext,ext]]).astype('float')

    
def make_dir(dir,overwrite_existing=False):
    if not os.path.exists(dir):
        os.makedirs(dir)
    elif overwrite_existing:
        filesToRemove = [os.path.join(dir,f) for f in os.listdir(dir)]
        for f in filesToRemove:
            os.remove(f) 
        
class RetinaArteryY(GrowthProcess):

    def __init__(self, handedness='L',Y_branching_factor=20.,T_branching_factor=18.,segment_fraction=1.):
        super().__init__()
        self.name = 'arteryY' #(was proc=0)
        self.type = 'artery'
        self.handedness = handedness
        self.Y_branching_factor = Y_branching_factor
        self.T_branching_factor = T_branching_factor
        self.segment_fraction = segment_fraction

    def get_branching_parameters(self,branch,segment=None,direction=None,gen=None):

        diameter = branch.tip_segment.diameters[-1]
        if diameter>50.:
            alpha, gamma = 1., 2.1
            alpha_sd, gamma_sd = 0.05, 0.05
            
        else:
            alpha, gamma = 0.8, 2.
            alpha_sd, gamma_sd = 0.02, 0.02
            
        alpha, gamma = np.random.normal(alpha,alpha_sd), np.random.normal(gamma,gamma_sd)
        alpha = np.clip(alpha,0.,0.99)
            
        return alpha, gamma
        
    def calculate_branching_distance(self,branch,segment=None,diameter=None):
            
        if diameter is None:
            if branch is None:
                return None
            if segment is None:
                segment = branch.tip_segment
            if segment is None:
                return None
            diameter = segment.diameters[-1]

        branching_factor = self.Y_branching_factor #18. # 18. #self.get_branching_factor(branch)
        if branch.daughter_count==0 and branch.order==0:
            mn, mx = 1., diameter*branching_factor
            branching_dist = np.random.uniform(mn,mx)
            #print('INIT BRANCING DIST:{}'.format(branching_dist))
            branching_dist = mx
        else:
            mn, sd = 1., diameter*branching_factor
            branching_dist = np.clip(np.random.normal(mn,sd),mn,None)
            
        # TEST ###
        branching_dist = diameter*branching_factor 
        # TEST ###

        return branching_dist
        
    def calculate_segment_length(self,branch,segment=None,params=None,diameter=None):  
        bd = self.calculate_branching_distance(branch,segment=segment)
        if bd is None:
            return 10.
        else:
            if bd==0.:
                breakpoint()
            return bd*self.segment_fraction # 10.
        
    def calculate_branching_diameters(self,branch,segment=None,direction=None,gen=None):
        
        diameters = super().calculate_branching_diameters(branch,segment=segment,direction=direction,gen=gen)
        #print('DIAMETERS:{}'.format(diameters))
        
        #if gen is not None:
        #    if gen['branching_type']==1: #'T'
        #        diameters[0] = branch.tip_segment.diameters[-1]
        #        diameters[1] = diameters[0] * 0.2
        #        if diameters[1]<100: # Set to optimising growth proc
        #            branch.growth_proc = 1
        
        #if self.handedness=='R':
        #    diameters[0],diameters[1] = diameters[1],diameters[0]

        return diameters   
        
    def calculate_branching_angles(self, branch):
        #a0,a1 = 30.,30. 
        #breakpoint()

        alpha,gamma = self.get_branching_parameters(branch)
        brad = np.arccos( (np.power((1+np.power(alpha,3)),4./3.) + np.power(alpha,4) - 1.) / (2.*np.square(alpha)*np.power(1.+np.power(alpha,3),2./3.)) )
        brad_1 = np.arccos( (np.power((1+np.power(alpha,3)),4./3.) + 1. - np.power(alpha,4)) / (2.*np.power(1.+np.power(alpha,3),2./3.)) )

        a1, a0 = np.rad2deg(brad_1), np.rad2deg(brad)
        a0 += np.random.normal(0.,5.)
        a1 += np.random.normal(0.,5.)
        
        # Negative angle values for right-handedness
        if self.handedness=='R':
            a0,a1 = -a0,-a1
        elif self.handedness=='N':
            if np.random.uniform()>0.5:
                a0,a1 = -a0,-a1
            
        # Probability of alternate branching pattern to that defined by handedness
        if np.random.uniform(0.,1.)>0.9:
            a0,a1 = -a0,-a1
         
        return a1, a0  
        
    def add_branch_here(self, branch, **kwargs):
        
        branch_length = branch.get_length()
        gen = [] #self._initialiser_generator()
        
        # Decide whether to add branch
        add_branch = False
        
        # Set branching distance if not yet defined
        if branch.interbranch_distance is None:
            self.set_interbranch_distance(branch)
        
        Tname = self.name.replace('Y','T')
        
        if 'segment' in kwargs.keys() and kwargs['segment'] is not None:
            segment = kwargs['segment']
        else:
            segment = branch.tip_segment
            
        last_T_branch_dist, last_T_node = branch.get_distance_from_last_branch(process_name=Tname, end_node=segment.end_node, return_node=True,parents=True)
        if segment!=branch.tip_segment:
            next_T_branch_dist, next_T_node = branch.get_distance_to_next_branch(process_name=Tname, start_node=segment.end_node, return_node=True,parents=True)
        else:
            next_T_branch_dist, next_T_node = None, None
            
        last_Y_branch_dist = branch.get_distance_from_last_branch(process_name=self.name,end_node=segment.end_node)
        if segment!=branch.tip_segment:
            next_Y_branch_dist, next_Y_node = branch.get_distance_to_next_branch(process_name=self.name, start_node=segment.end_node, return_node=True,parents=True)
        else:
            next_Y_branch_dist, next_Y_node = None, None
        
        add_T_branch = False
        
        cycle = -1
        min_t_cycle = -1
        if 'cycle' in kwargs.keys():
            cycle = kwargs['cycle']
            min_t_cycle = -1

        branching_dist = segment.diameters[-1] * self.T_branching_factor
        #print('Last T: {}, next T: {}, branching dist: {},tip:{}'.format(last_T_branch_dist,next_T_branch_dist,branching_dist,segment==branch.tip_segment))
        
        #if ( (last_T_branch_dist is None and next_T_branch_dist is None) or \
        #     (last_T_branch_dist is None and next_T_branch_dist>branching_dist) or \
        #     (next_T_branch_dist is None and last_T_branch_dist>branching_dist) or \
        #     (last_T_branch_dist is not None and next_T_branch_dist is not None and np.abs(next_T_branch_dist+last_T_branch_dist)>2*branching_dist) ) and \
        #     cycle>=min_t_cycle and \
        #     segment!=branch.tip_segment: 
        #    print('T distance criteria met!')
        if segment!=branch.tip_segment: 
           
           # and \            #(next_T_branch_dist is None or next_T_branch_dist>branching_dist) and \
           
            T_end_coord, T_length, T_diameters = None, None, None

            if False: #'endpoints' in kwargs.keys() and kwargs['endpoints'] is not None: # Use endpoints to create T branches

                endpoints = kwargs['endpoints']
                dists = arr([np.linalg.norm(x-segment.end_node.coords) for x in endpoints])
                max_dist = 1000.

                pnt = None

                if np.any(dists<max_dist):
                    #close_endpoints = endpoints[dists<max_dist]
                    inds = np.argsort(dists)
                    dists,endpoints = dists[inds],endpoints[inds]
                    close_endpoints = endpoints[dists<max_dist]
                    
                    ind = 0
                    while True:
                        pnt = geometry.point_line_closest(close_endpoints[ind],segment.start_node.coords,segment.end_node.coords,on_line=True,snap_to_vert=False)
                        if pnt is not None and norm(close_endpoints[ind]-pnt)>1e-12:
                            T_end_coord = close_endpoints[ind]
                            break
                        ind += 1
                        if ind>=len(close_endpoints):
                            break
                            
                    if pnt is not None and T_end_coord is not None:
                        add_T_branch = True
                        div_coord = pnt
                        
                        # Make sure division point doesn't lie on either node
                        if norm(div_coord-segment.start_node.coords)<1e-12:
                            ab = segment.end_node.coords - segment.start_node.coords
                            div_coord = segment.start_node.coords + ab*0.99
                        if norm(div_coord-segment.start_node.coords)<1e-12:
                            ab = segment.end_node.coords - segment.start_node.coords
                            div_coord = segment.start_node.coords + ab*0.01
                            
                        seg0 = segment
                        if norm(T_end_coord-div_coord)>1e-6:
                            proc = RetinaArteryT() # temporary instance
                            T_length = norm(T_end_coord-div_coord)
                            diam = proc.calculate_branching_diameters(branch,segment=segment)
                            T_diameters = arr([diam[1],diam[1]])
                        else:
                            add_T_branch = False

            else:
                # If previous T-branches have been defined
                if last_T_branch_dist is not None:
                    length, path = branch.get_distance_between_nodes(start_node=last_T_node,end_node=segment.end_node,return_path=True)
                    if len(path)>0:
                        #breakpoint()
                        lengths = np.cumsum([seg.get_length() for seg in path])
                        path = np.asarray(path)
                        if np.any(lengths>=branching_dist):
                            div_seg = np.asarray(path)[lengths>=branching_dist][0]
                            bounds = [div_seg.start_node.coords,div_seg.end_node.coords]
                            vec = bounds[1] - bounds[0]
                            vec_len = np.linalg.norm(vec)
                            vec = vec / np.linalg.norm(vec)
                            if np.any(lengths<branching_dist):
                                length_at_div_seg = lengths[lengths<branching_dist][-1]
                                fr = (branching_dist - length_at_div_seg) / div_seg.get_length()
                            else:
                                fr = 1.
                            
                            pts = div_seg.get_edge_coords()
                            if pts.shape[0]==2:
                                div_coord = div_seg.start_node.coords + fr*vec
                                
                            seg0 = div_seg
                            add_T_branch = True
                # If this is the first T-branch
                else:
                    length, path = branch.get_distance_between_nodes(return_path=True)
                    if len(path)>0:
                        lengths = np.cumsum([seg.get_length() for seg in path])
                        path = np.asarray(path)
                        rnd_distance = np.random.uniform(0.,branching_dist) # np.asarray(path)[lengths>=branching_dist][0]
                        
                        if np.any(lengths>=rnd_distance):
                            div_seg = np.asarray(path)[lengths>=rnd_distance][0]
                            bounds = [div_seg.start_node.coords,div_seg.end_node.coords]
                            vec = bounds[1] - bounds[0]
                            vec_len = np.linalg.norm(vec)
                            vec = vec / np.linalg.norm(vec)
                            
                            if np.any(lengths<rnd_distance):
                                length_at_div_seg = lengths[lengths<rnd_distance][-1]
                            else:
                                length_at_div_seg = 0.
                            fr = (rnd_distance - length_at_div_seg) / div_seg.get_length()
                            
                            pts = div_seg.get_edge_coords()
                            if pts.shape[0]==2:
                                div_coord = div_seg.start_node.coords + fr*vec
                                
                            seg0 = div_seg
                            add_T_branch = True
                            
            if T_diameters is not None and np.any(T_diameters<7.5):
                add_T_branch = False
            
            if add_T_branch:   
                new_T_location = div_coord 
                seg1 = seg0.divide(coord=div_coord)
                genT = self._initialiser_generator()
                genT['branching_type'] = 1
                if self.type=='artery':
                    genT['creation_process'] = RetinaArteryT()
                if self.type=='vein':
                    genT['creation_process'] = RetinaVeinT()
                genT['node'] = seg0.end_node
                genT['segment'] = seg0
                genT['end_point_coords'] = T_end_coord
                genT['length'] = T_length
                genT['diameters'] = T_diameters
                if T_end_coord is not None:
                    genT['fully_defined'] = True
                gen.append(genT)
        
        last_branch_dist = branch.get_distance_from_last_branch(process_name=self.name)
        prev_dir = branch.get_daughter_initialisation_branch_directions(process_name=self.name)
        branching_dist = branch.interbranch_distance #self.calculate_branching_distance(branch)
        if branching_dist is None:
            branching_dist = self.calculate_branching_distance(branch,segment=segment) #,diameter=diameter)
        
        # Whether to add a y-branch
        #breakpoint()
        #print(last_branch_dist,branching_dist,branch_length)
        #if last_branch_dist is not None and branching_dist is not None:
        #    if last_branch_dist>2*branching_dist:
        #        branch.status = 'inactive'
        #        return False, gen 
            
        min_branch_length = 0.
        branching_dist_rnd = np.random.uniform(min_branch_length,branching_dist)
        if ((last_branch_dist is not None and last_branch_dist>=branching_dist) or (last_branch_dist is None and branch_length>=branching_dist_rnd)) and segment==branch.tip_segment:
            genY = self._initialiser_generator()
            genY['branching_type'] = 0
            genY['creation_process'] = self
            gen.append(genY)
                
        return len(gen)>0, gen  
        
class RetinaArteryT(RetinaArteryY):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'arteryT' #(was proc=1)
        self.type = 'artery'

    def get_branching_parameters(self,branch,direction=None,gen=None):
        return 0.35, 2.1 # alpha, gamma 
    
    def calculate_segment_length(self,branch,segment=None,params=None,diameter=None):  
        bd = self.calculate_branching_distance(branch,segment=segment,diameter=diameter)
        if bd is None:
            return 10.
        else:
            if bd==0.:
                breakpoint()
            return bd*self.segment_fraction #0.2 # 10. 
            
    def calculate_branching_distance(self,branch,segment=None,diameter=None):
            
        if segment is None:
            segment = branch.tip_segment
            
        if diameter is None:
            if branch is None:
                return None
            if segment is None:
                return None

            diameter = segment.diameters[-1]

        branching_factor = self.T_branching_factor #18. # 18. #self.get_branching_factor(branch)
        if branch.daughter_count==0 and branch.order==0:
            mn, mx = 1., diameter*branching_factor
            branching_dist = np.random.uniform(mn,mx)
            #print('INIT BRANCING DIST:{}'.format(branching_dist))
        else:
            mn, sd = 1., diameter*branching_factor
            branching_dist = np.clip(np.random.normal(mn,sd),mn,None)
            
        # TEZST
        branching_dist = diameter*branching_factor

        return branching_dist    
        
    def calculate_branching_angles(self, branch, randomise=False):
        a0,a1 = -90.,None
        
        if randomise:
            a0 = np.random.normal(70.,10.)

        direction = branch.growth_direction
        midline_dir = self.eye.fovea_centre - self.eye.optic_disc_centre
        midline_dir = midline_dir / np.linalg.norm(midline_dir)
        
        fovea_direction = self.eye.fovea_centre - branch.tip_segment.end_node.coords
        fovea_direction = fovea_direction / np.linalg.norm(fovea_direction)
        
        new_direction1 = geometry.rotate_about_axis(direction,zaxis,90.)
        new_direction2 = geometry.rotate_about_axis(direction,zaxis,-90.)
        dt1 = np.dot(new_direction1,fovea_direction)
        dt2 = np.dot(new_direction2,fovea_direction)

        if dt1>dt2 and np.random.uniform()<0.9:
            a0 = -a0
        #else:
        #    a0 = a0
        
        if False: # point to midline
            dt = np.dot(direction,midline_dir)
            if dt>0:
                a0 = -90.
            else:
                a0 = 90.
            
        #a0 += np.random.normal(0.,3.)
        #a1 += np.random.normal(0.,3.)  
        return a1, a0                                                 

class RetinaVeinY(RetinaArteryY):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.name = 'veinY'             
        self.type = 'vein' 
        
class RetinaVeinT(RetinaArteryT):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.name = 'veinT' 
        self.type = 'vein' 

class RetinaSimulation(Simulation):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.artery_endpoints = None
        self.vein_endpoints = None
        
        if 'n_init_artery' in kwargs.keys():
            self.n_init_artery = kwargs['n_init_artery']
        else:
            self.n_init_artery = 2
        if 'n_init_vein' in kwargs.keys():
            self.n_init_vein = kwargs['n_init_vein']
        else:
            self.n_init_vein = 2
            
        if 'eye' in kwargs.keys() and kwargs['eye'] is not None:
            self.eye = kwargs['eye']
        else:
            self.eye = Eye()
            
        self.allow_midline_crossing = True
        
        self.tree.add_inherited_property(name='quadrant',dtype='int',type='branch')
        self.tree.add_inherited_property(name='handedness',dtype='int',type='branch')
        self.tree.add_inherited_property(name='growth_proc',dtype='int',type='branch')
        self.tree.add_inherited_property(name='branching_type',dtype='int',type='branch')
        self.tree.add_inherited_property(name='branching_type',dtype='int',type='segment')
        
        #self.tree.graph_params.append({'name':'branch','element_type':'branch','data_field':'id'})
        #self.tree.graph_params.append({'name':'branch_parent','element_type':'branch','data_field':'parent'})
        #self.tree.graph_params.append({'name':'quadrant','element_type':'branch','data_field':'quadrant'})
        #self.tree.graph_params.append({'name':'handedness','element_type':'branch','data_field':'handedness'})
        #self.tree.graph_params.append({'name':'growth_process','element_type':'branch','data_field':'growth_proc'})
        
        self.params.enable_tip_branching = True
        self.params.enable_apical_growth = True
        self.params.enable_lateral_growth = False
        self.params.enable_inline_branching = False

        self.params.inherit_length = True
        self.params.inherit_diameter_segment = True
        self.params.branch_diameter_rule = 'procedural'
        self.params.diameter_inheritance_factor = 0.9
        self.params.minimum_vessel_diameter = 80. # 7.5 # um
        self.params.minimum_vessel_diameter_branching_behaviour = 'stop' #'clip'
        self.params.maximum_vessel_order_branching_behaviour = 'stop'#'stop'
        self.params.minimum_vessel_diameter_growth_behaviour = 'stop'
        self.params.maximum_vessel_order = 3
        
        # SIMULATION PARAMS
        self.params.branch_pattern = 'alternate'
        self.params.branch_pattern_initial_direction = None

        # Branch apical growth parameters
        self.params.segment_length_random_frac = 0.1 # Fraction of length to set normal stdev to
        self.params.diameter_rand_frac = 0.0
        self.params.branching_angle = 50.
        self.params.branching_angle_sd = 0.
        self.params.deflect_lead_branch = True
            
        self.params.segment_bias = None #{ 'current':1.,
#                                     'same_vessel_avoidance':1.,
                                        #     }
        self.params.branching_bias = None #{'pattern':1.,
#                                      #'same_vessel_avoidance':1.
                                      #}
        self.params.vessel_avoidance_padding = 10.
        self.params.current_segment_bias_angle = 2. #0.1 #deg Angular size of prior (was 1.0) Path wobbliness
        self.params.max_segment_angle_per_step = 90.
        self.params.max_segment_sampling_angle = 20.
        self.params.branch_pattern_bias_angle = 10. # deg
        self.params.maximum_diameter_for_node_attraction = 20. # um
        self.params.maximum_diameter_for_connection = 15.  
        self.params.minimum_diameter_for_avoidance = 10.
        self.params.branching_factor = 18. #15. # Branching factors - multipled by diameter to give branching distance
        self.params.tortuosity_factor = 0.02
        
        self.params.branch_direction_function = self.branch_direction
        
        if 'export_surface_plot' in kwargs.keys():
            export_surface_plot = kwargs['export_surface_plot']
        else:
            export_surface_plot = False
        if 'plot' in kwargs.keys():
            plot = kwargs['plot']
        else:
            plot = False
        if 'plot_path' in kwargs.keys():
            plot_path = kwargs['plot_path']
        else:
            plot_path = None
            
        surface_type = 'flat' # 'layered'
        self.surface, self.surface_normals, self.surface_xv, self.surface_yv, self.surface_vals, self.seg_vals = self.create_retina_surface(eye=self.eye,domain=self.domain,plot=plot, surface_type=surface_type,export_surface_plot=export_surface_plot,plot_path=plot_path) # 'layered'
        self.generate_endpoints()
        
    def calculate_normals(self,surface,dx=1.,dy=1.):
    
        nx,ny = surface.shape[0],surface.shape[1]
        
        normals = np.zeros([nx,ny,3])
        
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                ndy = arr([0.,dy,surface[i,j+1] - surface[i,j-1]])
                ndx = arr([dx,0.,surface[i+1,j] - surface[i-1,j]])
                normals[i,j,:] = np.cross(ndx,ndy)
                normals[i,j,:] = normals[i,j,:] / np.linalg.norm(normals[i,j,:])
                if i==1:
                    normals[0,j,:] = normals[i,j,:]
                elif i==nx-2:
                    normals[-1,j,:] = normals[i,j,:]   
                if j==1:
                    normals[i,0,:] = normals[i,j,:]
                elif j==ny-2:
                    normals[i,-1,:] = normals[i,j,:]
        return normals  
        
    def create_raised_indent_surface(self,xv,yv,dx,normal=arr([0.,0.,1.]),fovea_centre=arr([0.,0.,0.]),nr=20,nproj=1001,fw=50.,shoulder_height=[100.,100.],shoulder_sd=[100.,100.],boundary_height=[0.,0.],flatness=0.2,fovea_depth=220.,raised=False):
    
        if not raised:
            coords = fovea_sim.fovea_sim(nr=nr,shoulder_height=shoulder_height,fw=fw,dr=fovea_depth,flatness=flatness,boundary_height=boundary_height,shoulder_sd=shoulder_sd,plot=False)
        else:
            coords = fovea_sim.fovea_sim_raised(nr=nr,nproj=nproj,fw=fw,dr=fovea_depth,flatness=flatness,boundary_height=boundary_height,plot=False)
        
        rad_dist = arr([np.linalg.norm(x[0:2]) for x in coords])
        width = fw/2. + np.max(shoulder_sd)*3
        blend = rad_dist / width
        blend[rad_dist<=width] = 1
        blend[rad_dist>width] = 0
        blend_mask = griddata(coords[:,0:2], blend, (xv,yv), method='nearest')
        
        from scipy.ndimage import gaussian_filter
        sigma = np.max([1,int(np.ceil(np.max(shoulder_sd)/dx))])
        blend_mask = gaussian_filter(blend_mask, sigma=sigma)

        # Align with normal
        coords = fovea_sim.align_with_normal(coords,arr([normal[0],normal[1],normal[2]]))
        # Translate to fovea centre
        coords[:,0:2] += [self.eye.fovea_centre[0],self.eye.fovea_centre[1]]
        
        # Regrid
        fovea_surf = griddata(coords[:,0:2], coords[:,2], (xv,yv), method='nearest')
        blend_mask = griddata(coords[:,0:2], blend, (xv,yv), method='nearest') 
        
        return fovea_surf, blend_mask, coords
        
    def create_retina_surface(self, domain=None, plot=False,surface_type='layered',export_surface_plot=False,plot_path=None,eye=None):

        # Domain size
        if domain is None: # um
            domain = DOMAIN #arr([[-5000.,5000.],[-5000.,5000.],[-5000.,5000.]])

        if surface_type=='layered':
            rs = retina_surface.RetinaSurface(domain=domain,eye=eye)
            plot_file = self.create_filename(-1,'retina_surface_'+self.prefix,'png')
            plot_file = join(plot_path,plot_file)
            #breakpoint()
            vessel_surf, normals, xv, yv, vals, seg_vals = rs.create_retina_surface(plot=plot,plot_file=plot_file)
            self.surfaces = rs.surfaces
            #breakpoint()
        else: # Three flat surfaces separated by 100 um
            # Grid params
            n = 500
            nx,ny = n,n
            dx = (domain[0,1]-domain[0,0])/float(nx)
            dy = (domain[1,1]-domain[1,0])/float(ny)
            x = np.linspace(domain[0,0],domain[0,1],nx)
            y = np.linspace(domain[1,0],domain[1,1],ny)
            xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
            vessel_surf = arr([xv.flatten(),yv.flatten(),np.zeros(n*n)]).transpose()
            self.surfaces = [vessel_surf[:,2].copy()+100.,vessel_surf[:,2],vessel_surf[:,2].copy()-100.]
            normals = np.zeros([nx,ny,3])
            normals[:,:,2] = 1.
            vals = arr([1,2])
            seg_vals = vals.copy()
            vessel_surf = np.reshape(vessel_surf[:,2],[nx,ny])

        return vessel_surf, normals, xv, yv, vals, seg_vals

    def generate_endpoints(self):
        
        if False:
            nf = 250
            endpoints_2d = boid_seed_generator(domain=arr([[0.,1.],[0.,1.]]),plot_data=False,nf=nf,nt=100,edge_repulsion=True,repul_strength=0.01,edge_repul_strength=0.0007)        
            endpoints = [] #np.zeros([endpoints_2d.shape[0],3])
            for i,x in enumerate(endpoints_2d):
                if np.all(x>=0.) and np.all(x<=1.):
                    coord = arr([x[0],x[1],0.])
                    coord[0:2] = (coord[0:2] * (self.domain[0:2,1] - self.domain[0:2,0])) + self.domain[0:2,0]
                    coord = self.snap_to_surface(coord)
                    endpoints.append(arr(coord))
                    
            endpoints = arr(endpoints)
            nf = int(np.floor(endpoints.shape[0]/2.))
            self.artery_endpoints = endpoints[:nf]
            self.artery_endpoints_occupied = np.zeros(self.artery_endpoints.shape[0])
            self.vein_endpoints = endpoints[nf:]
            self.vein_endpoints_occupied = np.zeros(self.vein_endpoints.shape[0])
        else:
            nf = 32
            x1, y1 = np.meshgrid(np.linspace(0., 1., nf), np.linspace(0., 1., nf))
            endpoints_2d = np.vstack([x1.flatten(),y1.flatten()]).transpose()
            endpoints = []
            for i,x in enumerate(endpoints_2d):
                if np.all(x>=0.) and np.all(x<=1.):
                    coord = arr([x[0],x[1],0.])
                    coord[0:2] = (coord[0:2] * (self.domain[0:2,1] - self.domain[0:2,0])) + self.domain[0:2,0]
                    coord = self.snap_to_surface(coord)
                    endpoints.append(arr(coord))
                    
            dx,dy = (self.domain[0,1] - self.domain[0,0])/float(nf) , (self.domain[1,1] - self.domain[1,0])/float(nf)

            endpoints = arr(endpoints)
            nf = int(np.floor(endpoints.shape[0]/2.))
            self.artery_endpoints = endpoints
            self.artery_endpoints_occupied = np.zeros(self.artery_endpoints.shape[0])
            endpoints_v = endpoints.copy()
            endpoints_v[:,0] += dx/2.
            endpoints_v[:,1] += dy/2.
            self.vein_endpoints = endpoints_v
            self.vein_endpoints_occupied = np.zeros(self.vein_endpoints.shape[0])
            
    def get_branching_parameters(self,branch,segment=None,direction=None):
    
        breakpoint()

        if branch.growth_proc==0:
            return 1., 2.1
        elif branch.growth_proc==1:
            return 0.6, 2.1
        else:
            return 0.6, 2.1

        calpha = 0.6
        if segment is None:
            segment = branch.tip_segment
        d0 = segment.diameters[-1]

        if branch.category=='artery':
            if d0<50.:
                alpha = np.clip(0.6+np.random(0.,0.02),None,0.99) # 0.8
                gamma = 2.1+np.random(0.,0.1)
            else: # Large artery
                alpha = np.clip(0.95+np.random(0.,0.02),None,0.99) #calpha # y-shape with alpha=0.99
                gamma = 2.7+np.random(0.,0.1)
        elif branch.category=='vein': # vein
            if d0<50.:
                alpha = np.clip(0.8+np.random(0.,0.02),None,0.99)
                gamma = 2.7+np.random(0.,0.1)
            else: # Large vein
                alpha = np.clip(0.95+np.random(0.,0.02),None,0.99)
                gamma = 2.1+np.random(0.,0.1)
            
        return alpha, gamma 
        
    #def get_branching_factor(self,branch):
    #    if branch.growth_proc==0:
    #        return self.params.branching_factor
    #    else:
    #        return self.params.branching_factor
        
    #def calculate_branching_distance(self,branch,segment=None,diameter=None):
    #       
    #    if diameter is None:
    #        if branch is None:
    #            return None
    #        if segment is None:
    #            segment = branch.tip_segment                
    #        if segment is None:
    #            return None
    #        diameter = segment.diameters[-1]
    #        
    #    if branch is not None:
    #        # First branch
    #        if branch.daughter_count==0 and branch.order==0:
    #            branching_factor = self.get_branching_factor(branch)
    #            branching_dist = np.random.uniform(10.,diameter*branching_factor*0.3)
    #        else:            
    #            branching_factor = self.get_branching_factor(branch)
    #            branching_dist = diameter * branching_factor
    #    else:
    #        branching_dist = np.random.normal(dfd/3.,250.) / 4. # 1800.
    #
    #    return branching_dist        
        
    def branch_direction(self,prev_dir,ni,bi,params=None):
    
        if len(prev_dir)>1:
            direction = prev_dir[-2]
        else:
            import pdb
            pdb.set_trace()
            direction = geometry.rotate_about_axis(prev_dir[-1],zaxis,180.)
        return direction 
        
    def surface_index_to_coord(self, inds, dims=None):
     
        domain = self.domain[0:2]
        if dims is None:
            dims = self.surface.shape.astype('float')
        else:
            dims = dims.astype('float')
        norm_coord = inds / dims
        return norm_coord*(domain[:,1]-domain[:,0]) + domain[:,0]
        
    def coord_to_surface_index(self, coords):
     
        domain = self.domain
        norm_coord = (coords - domain[:,0]) / (domain[:,1]-domain[:,0])
        return (norm_coord[0:2] * self.surface.shape).astype('int')
        
    def get_nearest_surface_normal(self,coords,surface_normals=None):
    
        x,y = self.coord_to_surface_index(coords)
        x = np.clip(x,0,self.surface.shape[0]-1)
        y = np.clip(y,0,self.surface.shape[1]-1)
        if surface_normals is None:
            return self.surface_normals[x,y]
        else:
            return surface_normals[x,y]
            
    def plot_surface(self, xv, yv, surface, normals=None, ax=None, show=True, vl=200.,skip=10,plot_surface=True):
    
        if ax is None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, projection='3d')
        
        if plot_surface:    
            ax.plot_surface(xv, yv, surface,cmap=cm.coolwarm)
        
        if normals is not None:
            for i,x in enumerate(xv[:,0]):
                for j,y in enumerate(yv[0,:]):
                    if i%skip==0 and j%skip==0:
                        ax.plot([x,x+normals[i,j,0]*vl], [y,y+normals[i,j,1]*vl], [surface[i,j],surface[i,j]+normals[i,j,2]*vl])
        if show:
            plt.show()
        return ax
        
    def below_midline(self,coords):
    
        A, B = self.eye.optic_disc_centre[0:2], self.eye.fovea_centre[0:2]
        A,B = A[0:2], B[0:2]
        pos = (B[0]-A[0]) * (coords[1]-A[1]) - (B[1]-A[1]) * (coords[0]-A[0])
        if pos>0:
            return True
        else:
            return False
        
    def get_quadrant(self, coords):
    
        # Find which quadrant the coordinate lies in, defined by the midline connecting the optic nerve / fovea
        
        #position = sign((Bx - Ax) * (Y - Ay) - (By - Ay) * (X - Ax))
        A, B = self.eye.optic_disc_centre[0:2], self.eye.fovea_centre[0:2]
        A,B = A[0:2], B[0:2]
        pos = (B[0]-A[0]) * (coords[1]-A[1]) - (B[1]-A[1]) * (coords[0]-A[0])
        if pos>0:
            below_midline = True
        else:
            below_midline = False
        
        A = self.eye.optic_disc_centre[0:2]
        A = A[0:2]
        if False: # orthogonal to midline
            direction = arr([self.eye.fovea_centre[1],self.eye.fovea_centre[0]]) - arr([self.eye.optic_disc_centre[1],self.eye.optic_disc_centre[0]])
        else:
            direction = arr([0.,1.,0.])
        B = self.eye.optic_disc_centre + direction
        B = B[0:2]
        pos = (B[0]-A[0]) * (coords[1]-A[1]) - (B[1]-A[1]) * (coords[0]-A[0])
        if pos>0:
            left_of_optic_nerve = True
        else:
            left_of_optic_nerve = False
            
        if below_midline: # and left_of_optic_nerve:
            quad = 0
        else:
            quad = 2
        #elif below_midline and not left_of_optic_nerve:
        #    quad = 1
        #elif not below_midline and left_of_optic_nerve:
        #    quad = 2
        #elif not below_midline and not left_of_optic_nerve:
        #    quad = 3
            
        return quad
        
    def inside_optic_disc(self, coord):

        if norm(self.eye.optic_disc_centre[0:2]-coord[0:2])>self.eye.optic_disc_radius:
            return False
        else:
            return True
        
    def create_branch_callback(self,branch,**kwargs):
        if branch is None:
            return branch
            
        if 'gen' in kwargs.keys():
            gen = kwargs['gen']
        else:
            gen = None
            
        # Fix the branch quadrant of the branching order>1 or the branch has >1 branches
        if branch.quadrant==-1:
        
            set_quad = False
            
            tip_node = branch.get_tip_node()
            if tip_node is not None:
                #breakpoint()
                end_coord = tip_node.coords
                odc = self.snap_to_surface(self.eye.optic_disc_centre)
                dist = np.linalg.norm(end_coord - odc)
                
                #if gen is not None:
                #    if gen['branching_type']==1: # T
                #        set_quad = True # All t-branches are constrianed to the quadrant they a spawned in

                if branch.creation_process is not None and branch.creation_process.name in ['arteryT','veinT']:
                    set_quad = True
                        
                if set_quad or \
                   ( (branch.order>=1 or branch.daughter_count>=2) and \
                     (dist>2*self.eye.optic_disc_radius) ):
                
                    #print('OD:',dist,5*self.eye.optic_disc_radius)
                    quad = self.get_quadrant(end_coord)
                    branch.quadrant = quad
                
        # Deal with parent
        if branch.parent is not None and branch.parent.quadrant==-1:
            parent = branch.parent
            set_quad = False
            
            end_coord = parent.get_tip_node().coords
            odc = self.snap_to_surface(self.eye.optic_disc_centre)
            dist = np.linalg.norm(end_coord - odc)
            
            #if gen is not None:
            #    if gen['branching_type']==1: # T
            #        set_quad = True # All t-branches are constrained to the quadrant they a spawned in

            if parent.creation_process is not None and parent.creation_process.name in ['arteryT','veinT']:
                set_quad = True
                    
            if set_quad or \
               ( (parent.order>=1 or parent.daughter_count>=2) and \
                 (dist>self.eye.optic_disc_radius) ):
            
                #print('OD:',dist,self.eye.optic_disc_radius)
                quad = self.get_quadrant(end_coord)
                parent.quadrant = quad
                
        #if branch.order>2:
        #    branch.status = 'inactive'
        #if branch.tip_segment.diameters[-1]<10:
        #    branch.status = 'inactive'
        
        #if branch.tip_segment.diameters[-1]<12:
        #    import pdb
        #    pdb.set_trace()
                
        return branch
        
    def set_branch_quadrant(self, branch, quadrant):
    
        branch.quadrant = quadrant
        if quadrant==0 or quadrant==2:
            branch.handedness = 'R'
        else:
            branch.handedness = 'L'
        
    def add_segment_to_branch_callback(self,branch,new_segment,**kwargs):
    
        if branch is None or branch.status!='active':
            return branch
        # Fix the branch quadrant of the branching order>1 or the branch has >1 branches
        if branch.quadrant==-1 and (branch.order>1 or len(branch.get_branching_nodes())>=2):
            end_coord = branch.get_tip_node().coords
            odc = self.snap_to_surface(self.eye.optic_disc_centre)
            dist = np.linalg.norm(end_coord - odc)
            #print('OD:',dist,5*self.eye.optic_disc_radius)
            if dist>2*self.eye.optic_disc_radius: 
                quad = self.get_quadrant(end_coord)
                self.set_branch_quadrant(branch, quad)
                
        if 'realign' in kwargs.keys():
            realign = kwargs['realign']
            if len(realign)>0:         
                seg = branch.tip_segment
                start_coord = seg.coords[0]
                seg_len = np.linalg.norm(seg.coords[1]-seg.coords[0])
                new_end_coord = start_coord + seg_len*realign[0][0]  
                end_node = seg.end_node  
                end_node.coords = new_end_coord
                seg.coords[-1] = new_end_coord 
                seg.initial_direction = realign[0][0] 
                seg.direction = realign[0][0]
                seg.directions = [realign[0][0]]
                seg.normals = [realign[0][1]]
                seg.initial_normal = realign[0][1]
                import pdb; pdb.set_trace()
                
        #coll,coll_seg,coll_pnt = self.tree.segment_collision_check(new_segment)
        #if coll:
        #    breakpoint()
                
        return branch
        
    def find_domain_intersection(self,start_coord,end_coord,**kwargs):
        d_coord = super().find_domain_intersection(start_coord,end_coord,**kwargs)

        # Check for crossing midline
        A, B = self.eye.optic_disc_centre[0:2], self.eye.fovea_centre[0:2]
        A,B = A[0:2], B[0:2]
        midline_dir = B - A
        pad = 200000. #* 1e-6
        midline_dir = midline_dir / np.linalg.norm(midline_dir)
        collision,m_coord = geometry.segments_intersect_2d( start_coord[0:2], #segment.start_node.coords[0:2],
                                                            end_coord[0:2],
                                                            A-midline_dir*pad,
                                                            A+midline_dir*pad)
        if m_coord is not None:
            return self.snap_to_surface(m_coord)
        elif d_coord is not None:
            return self.snap_to_surface(d_coord)
        else:
            return None
        
    def inside_domain(self,coord,**kwargs):
    
        # Coord is the proposed new segment end coordinate
        if not super().inside_domain(coord,**kwargs):
            return False

        branch = None
        end_coord = coord
        start_coord = None
        segment = None
        if 'segment' in kwargs.keys():
            segment = kwargs['segment']
            if segment is not None:
                # Test which quadrant the segment is in
                start_coord = segment.get_start_coords()
                quad_start = self.get_quadrant(start_coord)
                quad_end = self.get_quadrant(coord)
                branch = segment.branch
        if 'start_coord' in kwargs.keys() and 'branch' in kwargs.keys():
            start_coord, branch = kwargs['start_coord'],kwargs['branch']
            if start_coord is not None and branch is not None:
                # Test which quadrant the segment is in
                quad_start = self.get_quadrant(start_coord)
                quad_end = self.get_quadrant(coord)
        if branch is None: # No matches... Can happen during initialisation
            return True

        # Check if in the correct quadrant
        #if quad_end!=branch.quadrant and branch.quadrant!=-1:
        #    return False
        ch1,ch2 = self.below_midline(start_coord),self.below_midline(end_coord)
        in1,in2 = self.inside_optic_disc(start_coord),self.inside_optic_disc(end_coord)
        if (ch1!=ch2) and not in1 and not in2:
            #breakpoint()
            #if branch.order>1 or len(branch.daughters)>2: #breakpoint()
            if not self.allow_midline_crossing:
                print('Midline crossed',len(branch.daughters))
                return False

        # Check for crossing midline
        if False:
            A, B = self.eye.optic_disc_centre[0:2], self.eye.fovea_centre[0:2]
            A,B = A[0:2], B[0:2]
            midline_dir = B - A
            pad = 20000. #* 1e-6
            midline_dir = midline_dir / np.linalg.norm(midline_dir)
            collision,_ = geometry.segments_intersect_2d( start_coord[0:2],end_coord[0:2],A-midline_dir*pad,A+midline_dir*pad)
            #print(f'Midline collision: {collision}. {start_coord[0:2]}, {end_coord[0:2]}')                                    
            #if (start_coord[1]<0. and end_coord[1]>0.) or (start_coord[1]>0. and end_coord[1]<0.):
            #    breakpoint()
            if collision and branch.order>1:
                #breakpoint()
                return False
            #if collision and branch.order>2:
            #    return False

        return True
        
    def on_segment_collision(self, segment, coll_seg, intersection):
    
        # Decide what to do in the event of collisions
        return
        
        branch1 = segment.branch
        d1 = segment.diameters[-1]
        d2 = coll_seg.diameters[-1]
        
        #breakpoint()
        
        # Do nothing if inside optic disc
        if self.inside_optic_disc(segment.end_node.coords) or self.inside_optic_disc(intersection):
            return
        
        if coll_seg.branch.category==segment.branch.category:

            # Both are tip segments - allow the larger to continue
            if coll_seg==coll_seg.branch.tip_segment and segment==segment.branch.tip_segment:
                if d1<=d2:
                    branch1.set_property('status','inactive')
                    branch1.remove_segment(segment)
                    segment = None
                else:
                    branch2 = coll_seg.branch
                    branch2.set_property('status','inactive')
                    branch2.remove_segment(coll_seg)
            elif segment==segment.branch.tip_segment and coll_seg!=coll_seg.branch.tip_segment and d1>d2:
                # If segment crosses a smaller segment, remove the whole donwstream segment
                #breakpoint()
                branch2 = coll_seg.branch
                branch2.set_property('status','inactive')
                branch2.tree.remove_segment(coll_seg,propogate=True)
            else:
                branch1.set_property('status','inactive')
                branch1.remove_segment(segment)
                segment = None

        # If the collision is between two different category segments...
        elif False:
        #else:
            # If crossing from artery-vein is not almost 90deg, realign
            v1dir = (v1['end_coord']-v1['start_coord'])
            v1len = np.linalg.norm(v1dir)
            v1dir = v1dir / v1len
            v2dir = (v2['end_coord']-v2['start_coord'])
            v2len = np.linalg.norm(v2dir)
            v2dir = v2dir / v2len
            int_angle = np.abs(np.rad2deg(np.arccos(np.dot(v1dir,v2dir))))
            if int_angle>90.:
                int_angle = int_angle - 90.
            if int_angle<50.: # and v1['radius'] is not None and v2['radius'] is not None and v1['radius']<v2['radius']:
                new_dir1 = geometry.rotate_about_axis(v2dir,zaxis,90.)
                new_dir2 = geometry.rotate_about_axis(v2dir,zaxis,-90.)
                angleInd = np.argmin( [np.abs(np.arccos(np.dot(v1dir,new_dir1))),np.abs(np.arccos(np.dot(v2dir,new_dir1)))])
                if angleInd==0:
                    new_dir = new_dir1
                    if segment is not None:
                        new_norm = geometry.rotate_about_axis(segment.normal,zaxis,90.)
                    else:  
                        new_norm = xaxis
                else:
                    new_dir = new_dir2
                    if segment is not None:
                        new_norm = geometry.rotate_about_axis(segment.normal,zaxis,-90.)
                    else:  
                        new_norm = xaxis
                realign.append([new_dir,new_norm])
        else: # Allow vein-artery collisions only if both vessels are large
            coll_diameter = 60.
            if d1<coll_diameter or d2<coll_diameter:
                branch1.set_property('status','inactive')
                branch1.remove_segment(segment)
                segment = None
        
    def plot(self, figure=None,show=True):
        if figure is None:
            fig = plt.figure()
        for seg in self.tree.segments:
            c0,c1 = seg.start_node.coords,seg.end_node.coords
            plt.plot([c0[0],c1[0]],[c0[1],c1[1]],c='black')
        if show:
            plt.show()
        
    def calculate_world_branching_direction(self, branch, **kwargs):
    
        prev_dir = branch.get_daughter_initialisation_branch_directions()
        
        cur_dir = None
        proc = None        
        node = None
        
        if 'gen' in kwargs.keys():
            gen = kwargs['gen']
            if gen['node'] is not None:
                node = gen['node']
            if gen['segment'] is not None:
                segment = gen['segment']
                if node is None:
                    node = segment.end_node
                cur_dir = {'direction':segment.direction, 'normal':segment.normal}
            proc = gen['creation_process']
        if node is None:        
            node = branch.get_tip_node()
            cur_dir = branch.get_growth_direction()
        else:
            cur_dir = None
            if len(node.downstream_segments)>0:
                segs = [x for x in node.downstream_segments if x.branch==branch]
                if len(segs)>0:
                    seg = segs[0]
                    cur_dir = seg.get_direction()
            if cur_dir is None and len(node.upstream_segments)>0:
                seg = [x for x in node.upstream_segments if x.branch==branch]
                if len(segs)>0:
                    seg = segs[0]
                    cur_dir = seg.get_direction()
            if cur_dir is None:
                cur_dir = branch.get_growth_direction()
            if cur_dir is None:
                import pdb; pdb.set_trace()
        if proc is None:
            proc = branch.get_growth_process()
            
        coords = node.coords
        norm = self.get_nearest_surface_normal(coords)  # zaxis
        #new_dir = np.cross(cur_dir['direction'],norm)
        
        a1, a0 = proc.calculate_branching_angles(branch)
        
        #a0 = np.random.normal(a0,self.params.branching_angle_sd)
        norm = zaxis
        direction = geometry.rotate_about_axis(cur_dir['direction'],norm,a0)
                
        if proc.name in ['arteryT','veinT']:
            if np.abs(np.dot(direction,cur_dir['direction']))>0.1:
                #print(a1,a0,np.abs(np.dot(direction,cur_dir['direction'])))
                #import pdb; pdb.set_trace()
                pass
        
        #a1 = np.random.normal(a1,self.params.branching_angle_sd)
        if a1 is not None:
            ongoing_direction = geometry.rotate_about_axis(cur_dir['direction'],norm,-a1)
            ongoing_norm = norm
        else:
            ongoing_direction, ongoing_norm = None, None
        #print('World normal:{}, cur dir:{}, dir:{}. ongoing dir:{}'.format(norm,cur_dir,direction,ongoing_direction))
        #print(direction,norm,ongoing_direction)
        
        return direction, norm, [a1,a0], ongoing_direction, ongoing_norm      
        
    def calculate_branching_diameters(self,branch,segment=None,direction=None,gen=None):
        
        diameters = super().calculate_branching_diameters(branch,segment=segment,direction=direction,gen=gen)
        
        if segment is None:
            segment = branch.tip_segment
        
        if gen is not None:
            if gen['branching_type']==1: #'T'
                diameters[0] = branch.tip_segment.diameters[-1]
                diameters[1] = diameters[0] * 0.2
                if diameters[1]<100: # Set to optimising growth proc
                    branch.growth_proc = 1
        
        if branch.handedness=='R':
            diameters[0],diameters[1] = diameters[1],diameters[0]

        return diameters
        
    def snap_to_surface(self,coords):
        if len(coords)==2:
            coords = arr(np.concatenate([coords,[0.]]))
        xe,ye = self.coord_to_surface_index(coords)
        xe = np.clip(xe,0,self.surface.shape[0]-1)
        ye = np.clip(ye,0,self.surface.shape[1]-1)
        coords[2] = self.surface[xe,ye]
        return coords
        
    def calculate_branch_end_node_coords(self,start_coord,direction,length):
        end_coord = start_coord + direction*length
        return self.snap_to_surface(end_coord)
        
    def sim_vessel(self,angle_range=[-30,30],length=1.,vtype='artery',handedness='L',direction='down',diameter=None,avoid=None,avoid_angle=10.):
    
        if diameter is None:
            diameter = self.eye.adiameter
    
        #rndx = np.random.uniform(-angle_dev,angle_dev)
        while True:
            angle = np.deg2rad(np.random.uniform(angle_range[0],angle_range[1]))
            if avoid is not None:
                if np.rad2deg(np.abs(angle-avoid[0]))>avoid_angle:
                    break
            else:
                break
            
        if direction=='down':
            a0dir = arr([np.sin(angle),np.cos(angle),0.])
        elif direction=='up':
            a0dir = arr([np.sin(angle),-np.cos(angle),0.])
        elif direction=='left lower':
            angle = np.abs(angle)
            a0dir = arr([np.cos(angle),np.sin(angle),0.])
        elif direction=='left upper':
            angle = -np.abs(angle)
            a0dir = arr([np.cos(angle),np.sin(angle),0.])
        elif direction=='right':
            angle = np.abs(angle)
            a0dir = arr([-np.cos(angle),np.sin(angle),0.])
        else:
            print('Invalid direction: {}'.format(direction))
            return
        a0loc = self.eye.optic_disc_centre + self.eye.optic_disc_radius*0.0001*a0dir
        a0loc = self.snap_to_surface(a0loc)
        end_coord = self.calculate_branch_end_node_coords(a0loc,a0dir,length)
        if vtype=='artery':
            gp = RetinaArteryY(handedness=handedness,T_branching_factor=self.eye.T_branching_factor,Y_branching_factor=self.eye.Y_branching_factor,segment_fraction=self.eye.segment_fraction)
        else:
            gp = RetinaVeinY(handedness=handedness,T_branching_factor=self.eye.T_branching_factor,Y_branching_factor=self.eye.Y_branching_factor,segment_fraction=self.eye.segment_fraction)
        if direction=='down':
            quad = 0
        else:
            quad = 1
        branch = self.create_branch(diameter=diameter,direction=a0dir,length=length,start_coord=a0loc,category=vtype,growth_process=gp)  
        branch.quadrant = quad
        
        return branch,angle,a0loc,a0dir        
            

    def update_edge_paths(self,lock=False,remove_intermediate_nodes=False): 
        
        front = [self.tree.root_segment]
        visited = np.zeros(len(self.tree.segments))
        ninterp = 20
        
        if remove_intermediate_nodes:
            self.tree.remove_intermediate_nodes()
        
        for branch in self.tree.branches:
            if not branch.lock_paths:
                seg = branch.root_segment
                coords = []
                diameters, diameters_int = [],[]
                segs = []
                is_branch = [len(seg.end_node.upstream_segments)>1]
                count = 0
                while True:
                    coords.append(seg.start_node.coords)
                    segs.append(seg)
                    parents = seg.start_node.upstream_segments
                    if count==0 or len(parents)==0:
                        d0 = seg.diameter
                    else:
                        d0 = parents[0].diameter
                    d1 = seg.diameter
                    diameters.append(d0) #np.linspace(d0,d1,ninterp))
                    diameters_int.append(np.linspace(d0,d1,ninterp))
                    next_seg = [x for x in seg.end_node.downstream_segments if x.branch.id==seg.branch.id and x.status=='active']
                    is_branch.append(len(seg.end_node.downstream_segments)>1)
                    if len(next_seg)>0:
                        seg = next_seg[0]
                        count += 1
                    else:
                        coords.append(seg.end_node.coords)
                        diameters.append(d1)
                        break
                        
                #diameters_int = np.concatenate(np.asarray(diameters))
                diameters = arr(diameters)

                if True:
                    coords = arr(coords)
                    from vessel_sim.simplex_path import simplex_path,simplex_path_multi
                    points = np.zeros([(coords.shape[0]-1)*ninterp,3])

                    new_coords = coords.copy()
                    
                    start_dir,end_dir,fade,weight_mode = None,None,0.2,'linear'
                    feature_size = [np.mean(diameters)*10.]
                    displacement = 0.05 # 0.1=very curvy
                    points[:,0:2] = simplex_path_multi(coords[:,0:2],start_dir=start_dir,end_dir=end_dir,fade=fade,feature_size=[feature_size],npoints=ninterp,displacement=displacement,weight_mode=weight_mode) 

                    for i in range(points.shape[0]):
                        points[i] = self.snap_to_surface(arr([points[i,0],points[i,1],0.]))
                        
                else:
                    #diameters = arr(diameters).flatten()
                    mag = self.params.tortuosity_factor
                    if mag is None:
                        mag = 0.1
                    points = constrained_path(arr(coords),mag=mag,ninterp=ninterp)
                    
                for i in range(points.shape[0]):
                    points[i] = self.snap_to_surface(points[i])
                
                if False:
                    fig = plt.figure(figsize=[10,10])
                    ax = fig.add_subplot(111, projection='3d')
                    P = points
                    ax.plot(points[:,0],points[:,1],points[:,2],c='b')
                    ax.plot(coords[:,0],coords[:,1],coords[:,2],c='b')
                    #ax.plot(xint,yint,zint,c='r')
                    #ax.plot(midline[:,0],midline[:,1],midline[:,2],c='r')
                    plt.show()
                
                count = 0
                for i,seg in enumerate(segs):
                    crds = points[i*ninterp:(i*ninterp)+ninterp]
                    
                    segs[count].start_node.coords = crds[0]
                    segs[count].end_node.coords = crds[-1]


                    res = segs[count].set_edge_points(crds,diameters=diameters_int[i]) #*ninterp:(i*ninterp)+ninterp])
                    count += 1
                    
                if lock:
                    branch.lock_paths = True
                    
    def add_t_branch_to_segment(self, seg, length=2000.,direction=arr([1.,0.,0.]),fr=0.5,diameters=[50.,50.]):
    
        gp = RetinaVeinT()#handedness=handedness,T_branching_factor=self.eye.T_branching_factor,Y_branching_factor=self.eye.Y_branching_factor,segment_fraction=self.eye.segment_fraction)
        
        branch = seg.branch
        
        pts = seg.get_edge_coords()
        vec = seg.end_node.coords - seg.start_node.coords
        #vec /= np.linalg.norm(vec)

        div_coord = seg.start_node.coords + fr*vec #*length  
        new_T_location = div_coord 
        seg1 = seg.divide(coord=div_coord)
        genT = gp._initialiser_generator()
        genT['branching_type'] = 1
        genT['creation_process'] = RetinaArteryT()
        genT['node'] = seg.end_node
        genT['segment'] = seg
        T_end_coord = div_coord + direction*length
        if not self.inside_domain(T_end_coord,segment=seg):
            return
        
        genT['end_point_coords'] = T_end_coord
        genT['length'] = length
        genT['diameters'] = diameters
        if T_end_coord is not None:
            genT['fully_defined'] = True

        return self.add_branch_to_segment(seg,growth_process=[genT])
        
    def add_y_branch_to_segment(self, seg, length=2000.,direction=arr([1.,0.,0.]),fr=0.5, handedness=None):
    
        if handedness is None and seg.branch.growth_process is not None:
            handedness = seg.branch.growth_process.handedness
    
        gp = RetinaVeinY(handedness=handedness) #,T_branching_factor=self.eye.T_branching_factor,Y_branching_factor=self.eye.Y_branching_factor,segment_fraction=self.eye.segment_fraction)
        
        branch = seg.branch
        
        pts = seg.get_edge_coords()
        vec = seg.end_node.coords - seg.start_node.coords
        #vec /= np.linalg.norm(vec)

        div_coord = seg.start_node.coords + fr*vec #*length  
        new_T_location = div_coord 
        seg1 = seg.divide(coord=div_coord)
        genY = gp._initialiser_generator()
        genY['branching_type'] = 1

        genY['creation_process'] = RetinaArteryY(handedness=handedness)
        genY['node'] = seg.end_node
        genY['segment'] = seg
        T_end_coord = None
        diameters = [50.,50.] # um
        breakpoint()
        direction, norm, [a1,a0], ongoing_direction, ongoing_norm = self.calculate_world_branching_direction(branch)  
        T_end_coord = div_coord + direction*length   
        #breakpoint()
        genY['end_point_coords'] = T_end_coord
        genY['length'] = length
        genY['diameters'] = diameters
        if T_end_coord is not None:
            genY['fully_defined'] = True
            
        branch.next_segment_params = {'diameter':None,'angle':-a0,
                                      'world_direction':ongoing_direction,'world_normal':ongoing_norm,
                                      'branch_direction':None,'branch_normal':None}

        return self.add_branch_to_segment(seg,growth_process=[genY])

def save_surface(path,filename,sim):
    surface_file = join(path,filename.replace('.am','_surface.npz'))
    np.savez(surface_file,sim.surface_xv,sim.surface_yv,sim.surface,sim.surface_normals,sim.surfaces,sim.surface_vals,sim.seg_vals)
    
def simulate_cco_seed(prefix='',params=None,max_cycles=10,path=None,plot=False,dataPath=None,domain=None,eye=None,diameter=None):

    #print('Simulating CCO seed...')

    if path is None:
        path = r'/mnt/data2/retinasim/cco'
    
    make_dir(path)
    #make_dir(join(path,'graph'))
    #make_dir(join(path,'surface_plot'))
    
    create_graphs = True
    embed_graphs = False   
    allow_midline_crossing = True 

    np.random.seed()
    
    # Vessel initialisation params
    angle_dev = 0.1
    angle_range=[-10.,40.]    
    avoid_angle = 2.5
    length = 1.
    
    # Geometry params
    if domain is None:
        domain = DOMAIN
    domain_size = arr([x[1]-x[0] for x in domain])
    embed = False
    write_longitudinal = False

    store_graph,store_vessel_grid,store_mesh = True,True,False
    
    # Processing params
    proc_index = 0
    if prefix=='' and params is not None:
        if 'uid' in params.keys():
            prefix = params['uid']
        if 'proc_index' in params.keys():
            proc_index = params['proc_index']
    
    # Loop through quadrant (i.e. above or below macular) and vessel type (artery/vein)
    count = 0
    if eye is None:
        eye_geom = None
    else: 
        eye_geom = eye
    cylinders = []
    
    min_first_branch_length,max_first_branch_length = 0.,2500. # um
    first_branch_length = np.clip(np.random.uniform(min_first_branch_length-100,max_first_branch_length),1.,None)

    for quad in ['upper','lower']:
        for vessel in ['artery','vein']:

            plot_folder = join(path,'surface_plot')
            
            sim = RetinaSimulation(prefix=prefix,path=path,domain_size=domain_size,domain=domain,ignore_domain=False,domain_type='sphere',embed=False, 
                             store_graph=store_graph,store_vessel_grid=store_vessel_grid,store_mesh=store_mesh, 
                             initialise_folders=False,write_longitudinal=False,n_sampling_directions=11,verbose=0,max_cycles=max_cycles,
                             plot=True,export_surface_plot=True,plot_path=plot_folder,eye=eye_geom)
            sim.allow_midline_crossing = allow_midline_crossing
                             
            if count==0:
                eye_geom = sim.eye
            
            def _seed_vessel_branching(sim,handedness='R',vessel_type='artery',direction='down',avoid=None):
                #print(f'Init length: {first_branch_length}')
                cur_len = first_branch_length + np.clip(np.random.normal(0.,50.),min_first_branch_length,max_first_branch_length)
                if vessel_type=='artery':
                    diam = sim.eye.adiameter #+np.random.normal(0.,2.)
                else:
                    diam = sim.eye.vdiameter #+np.random.normal(0.,2.)
                vdata = sim.sim_vessel(vtype=vessel_type,direction=direction,length=cur_len,diameter=diam,handedness=handedness,avoid=avoid,angle_range=angle_range)
                br,vdata = vdata[0],vdata[1:]
                
                if handedness=='R':
                    opp_handedness = 'L'
                elif handedness=='L':
                    opp_handedness = 'R'
                
                # Manually add the first y-branch
                proc = br.get_growth_process()
                br.set_property('interbranch_distance',first_branch_length)
                print(f'First branch length {first_branch_length}')

                # Add in random additional branches
                def rnd_additional_branch(sim,ylim=0.1,xdir=1.,fr=None,direction='down',diameter=None):
                    # Fractional position along branch
                    if fr is None:
                        fr = np.random.uniform(0.01,0.8)
                    cdir = arr([xdir,np.random.uniform(0.,ylim),0.])
                    if direction=='up':
                        cdir[0:2] = -cdir[0:2]
                    if diameter is None:
                        diameter = np.random.normal(75.,15.)
                    _,new_branch = sim.add_t_branch_to_segment(br.root_segment,length=br.root_segment.get_length()*np.random.uniform(0.4,0.8),direction=cdir,fr=fr,diameters=[diameter,diameter])
                    new_branch.growth_process.handedness = 'N'
                
                if np.random.uniform()>0.5: #cur_len>1500. and np.random.uniform()>0.5:
                    xdir = 1.
                    rnd_additional_branch(sim,ylim=0.5,xdir=xdir,fr=np.random.uniform(0.01,0.1),direction=direction)
                if np.random.uniform()>0.5: #cur_len>1500. and np.random.uniform()>0.5:
                    xdir = 1.
                    rnd_additional_branch(sim,ylim=0.5,xdir=xdir,fr=np.random.uniform(0.5,1.0),direction=direction)
                
                if np.random.uniform()>0.3:  
                    xdir = -1.
                    rnd_additional_branch(sim,ylim=0.8,xdir=xdir,direction=direction)
                
                return vdata

            if quad=='lower':
                direction = 'down'
                if vessel=='artery':
                    handedness = 'L'
                    avoid = None
                    vdata_artery_lower = _seed_vessel_branching(sim,handedness='L',avoid=avoid,vessel_type=vessel,direction=direction)                    
                else:
                    handedness='L'
                    avoid = vdata_artery_lower
                    vdata_vein_lower = _seed_vessel_branching(sim,handedness='L',avoid=avoid,vessel_type=vessel,direction=direction)
                   
            elif quad=='upper':
                direction = 'up'
                if vessel=='artery':
                    handedness = 'R'
                    avoid = None
                    vdata_artery_upper = _seed_vessel_branching(sim,handedness='R',avoid=avoid,vessel_type=vessel,direction=direction) 
                else:
                    handedness = 'R'
                    avoid = vdata_artery_upper
                    vdata_vein_upper = _seed_vessel_branching(sim,handedness='R',avoid=avoid,vessel_type=vessel,direction=direction) 
     
            if params is not None:
                for k,v in zip(params.keys(),params.values()):
                   setattr(sim.params,k,v)

            sim.exit_on_no_change = True
            sim.params.minimum_vessel_diameter = 55. # um
            sim.max_cycles = max_cycles
    
            #print('Starting simulation {}...'.format(proc_index))
    
            if sim.max_cycles>0:
                try:
                    sim.run(parallel=False,branch_update=True,segment_update=False,write=False)
                    #sim.tree.remove_intermediate_nodes()
                    #sim.update_edge_paths(lock=True)
                    #breakpoint()
                    
                    gen = []
                    
                    #seg_change_in_cycle = sim.process_segment(sim.tree.root_segment) #.remote
                    
                    if False:
                        sim.max_cycles += 4
                        print('Adding T-branches')
                        sim.run(parallel=False,branch_update=False,segment_update=True,resume=True)
                    #print('Simulation {} complete'.format(proc_index))
                except KeyboardInterrupt:
                    print('Keyboard interrupt (proc {})'.format(proc_index))
                
            sim.tree.remove_intermediate_nodes()
            sim.tree.remove_orphan_nodes()
        
            # Ouput filename                
            filename = 'retina_{}_{}.am'.format(vessel,quad)
            #filename = sim.create_filename(sim.current_cycle,filename,'am',prefix=sim.prefix,longitudinal=sim.write_longitudinal)
            gfile = join(sim.gpath,filename)
            sg = sim.tree.write_amira(gfile)
            print('Graph written to {}'.format(gfile))
            
            if plot:
                graph = spatialgraph.SpatialGraph()
                graph.read(gfile)
                tp = graph.plot_graph(show=False)
                cyl = tp.cylinders_combined
                if vessel=='artery':
                    cyl.paint_uniform_color([1.,0.,0.])
                elif vessel=='vein':
                    cyl.paint_uniform_color([0.,0.,1.])
                cylinders.append(cyl)
            
            #txt_file = join(sim.gpath,filename.replace('.am','_discs_params.json'))
            #vcfg = VascularConfig(sim.eye,graph_artery_filename=filename,quad=quad,dataPath=dataPath)
            #vcfg.write(txt_file)

            save_surface(sim.gpath,filename,sim)
            count += 1
            
    eye_geom.save(join(sim.gpath,'retina_geometry.p'))
    
    if len(cylinders)>0:
        nc = 1000
        theta = np.linspace(0,2*np.pi,nc)
        
        pcd = o3d.geometry.PointCloud()
        circ = np.zeros([nc,3])
        circ[:,0] = (domain[0,1]-domain[0,0])*np.sin(theta) / 2.
        circ[:,1] = (domain[1,1]-domain[1,0])*np.cos(theta) / 2.
        pcd.points = o3d.utility.Vector3dVector(circ)
        pcd.paint_uniform_color([0, 0, 1])
        
        pcd1 = o3d.geometry.PointCloud()
        circ = np.zeros([nc,3])
        fovea_radius = 150.
        circ[:,0] = fovea_radius*np.sin(theta) + eye_geom.fovea_centre[0]
        circ[:,1] = fovea_radius*np.cos(theta) + eye_geom.fovea_centre[1]
        pcd1.points = o3d.utility.Vector3dVector(circ)
        pcd1.paint_uniform_color([0, 0, 1])
        
        o3d.visualization.draw_geometries(cylinders+[pcd,pcd1],mesh_show_wireframe=False)
    
    return gfile

    
def simulate_cco_seed_batch():

    path = r'/mnt/data2/retinasim/cco_batch'
    nsim = 100
    for i in range(nsim):
        path_cur = os.path.join(path,'sim_{}'.format(i))
        #prefix = 'sim_{}_'.format(i)
        simulate_cco_seed(path=path_cur)
    
if __name__=='__main__':

    if True:
        simulate_cco_seed(plot=True,max_cycles=10)
    if False:
        simulate_cco_seed_batch()
