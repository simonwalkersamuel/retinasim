import numpy as np

class Params(object):

    def __init__(self):
        self.enable_tip_branching = True
        self.enable_apical_growth = True
        self.enable_lateral_growth = False
        self.enable_inline_branching = False
        self.remove_stagnant_branches = False
        self.maximum_diameter_for_inline_branch = 50. # um
        self.update_graph_only = True
        self.inherit_length = True
        self.inherit_diameter_segment = True
        self.diameter_inheritance_factor = 0.9
        self.fixed_diameter_branch = None
        self.unit_diameter = 2. # um
        self.unit_length = 100. # um
        self.stop_growth_at_minimum_vessel_size = False
        self.minimum_vessel_diameter_branching_behaviour = 'stop' # or 'stop',clip'
        self.minimum_vessel_diameter = 10.
        self.maximum_vessel_order_branching_behaviour = 'stop'
        self.maximum_vessel_order = 5
        
        # SIMULATION PARAMS
        self.cycle_update = 10
        self.branch_pattern = 'alternate'
        self.branch_pattern_initial_direction = None
        self.branch_direction_function = None
        self.segment_length_function = None
        self.n_node_per_segment = 2
        self.branch_init_pattern = 'procedural'

        # Branch apical growth parameters
        self.segment_length_random_sd = 0. 
        self.segment_length_random_frac = 0. 
        self.diameter_rand_frac = 0.0
        self.branching_angle = 50.
        self.branching_angle_sd = 0.
        self.bisecting_plane = None
        self.minimum_branching_order_for_bisecting_plane = -1
        self.deflect_lead_branch = False
        self.use_growth_factor = True
            
        self.segment_bias = { 'current':1. }
        self.branching_bias = {'pattern':1.,'same_vessel_avoidance':1.}
        
        self.branching_alpha = 0.5 #1. #0.7 #0.5
        self.branching_gamma = 2.7
        self.vessel_avoidance_padding = 10.
        self.sampling_mode = 'probabilistic'
        self.current_segment_bias_angle = 2.
        self.max_segment_angle_per_step = 90.
        self.max_segment_sampling_angle = 20.
        self.branch_pattern_bias_angle = 10. # deg
        self.maximum_diameter_for_node_attraction = 20. # um
        self.maximum_diameter_for_connection = 15.  
        self.minimum_diameter_for_avoidance = 10.
        self.apical_growth_direction_sd = 0. #sim.unit_length*0.1
        self.first_branch_fractional_length = 10.
        self.add_branch_callback = None
        self.tortuosity_factor = 0.2
        self.branching_factor = 25. # Branching factors - multipled by diameter to give branching distance
        
    def get(self,key,default=None):
        return getattr(self,key,default)

class GrowthProcess(Params):

    def __init__(self):
        super().__init__()
        self.name = 'default'
        
    def add_branch_here(self, branch):
    
        last_branch_dist = branch.get_distance_from_last_branch()    
        prev_dir = branch.get_daughter_initialisation_branch_directions()
        branch_length = branch.get_length()
        
        gen = self._initialiser_generator()
        
        # Decide whether to add branch
        add_branch = False
        
        # Set branching distance if not yet defined
        if branch.interbranch_distance is None:
            self.set_interbranch_distance(branch)
            
        # If branching nodes are already present and its not the first branch
        if last_branch_dist is not None:
            add_branch = last_branch_dist>=branch.interbranch_distance
        else:
            # Decide to add a new branch if the current branch is longer than the pre-defined branch length
            add_branch = branch_length>=branch.interbranch_distance
            
        print('Add branch:{}'.format(add_branch))
        #if add_branch:
        #    import pdb
        #    pdb.set_trace()
            
        return add_branch, gen
        
    def calculate_branching_distance(self,branch,diameter=None):
            
        if diameter is None:
            if branch is None:
                return None
            if branch.tip_segment is None:
                return None
            diameter = branch.tip_segment.diameters[-1]
            
        branching_factor = self.get_branching_factor(branch)
        branching_dist = diameter * branching_factor
        
        #print('Calc branching dist: {}, f:{}'.format(branching_dist,branching_factor))

        return branching_dist
            
    def calculate_segment_length(self,branch,params=None,diameter=None):
    
        if branch is None:
            branching_dist = self.calculate_branching_distance(None,diameter=diameter)
            if branching_dist is None:
                return None
            else:
                return branching_dist
        else:
            # Define the interbranch distance for the current branch (if not possible, return)
            if branch.interbranch_distance is None:
                if not self.set_interbranch_distance(branch,diameter=diameter):
                    return None
                
        # Get the tip segment length since last branch (or root node)
        n_daughter = len(branch.daughters)
        if n_daughter>0:
            current_segment_length = branch.get_distance_from_last_branch()
        else:
            #print('calc seg len: using tip segment length')
            current_segment_length = branch.tip_segment.get_length()
        
        # Next segment length should be the remaining distance to make up the distance to the next branch point
        if current_segment_length is None:
            #import pdb
            #pdb.set_trace()
            return None
        next_length = branch.interbranch_distance - current_segment_length
        # If we're essentially *at* the branching distance, make the next length equal to the interbranch distance
        if next_length<=0.:
            next_length = branch.interbranch_distance
            
        # Subdivide the length by the number of required nodes (default is n_node_per_branch=2; other values may not work or give uneven node distributions)
        n_node_per_branch = self.n_node_per_segment
        # How many nodes remaining in the segment (will be 2 by default)
        rem_node = np.clip(n_node_per_branch - len(branch.segments)*2,2,n_node_per_branch)
        next_length = next_length / float(rem_node-1)
        
        # Limit the length to between th unit length and the interbranch distance
        next_length = np.clip(next_length,self.unit_length,branch.interbranch_distance)

        return next_length

    def set_interbranch_distance(self,branch,overwrite=False,diameter=None):
    
        if branch.interbranch_distance is not None and overwrite is False:
            return True 
            
        branching_dist = self.calculate_branching_distance(branch,diameter=diameter)
        if branching_dist is None:
            return False
            
        if self.segment_length_random_sd>0.:
            branching_dist += np.random.normal(0.,self.segment_length_random_sd)
        if self.segment_length_random_frac>0.:
            branching_dist += np.random.normal(0.,branching_dist*self.segment_length_random_frac)            
            
        branch.set_property('interbranch_distance',branching_dist) # np.random.normal(branching_dist,branching_dist_sd)        
        return True        
        
    def calculate_branching_diameters(self,branch,segment=None,direction=None,gen=None):
        
        if segment is None:
            segment = branch.tip_segment
        d0 = segment.diameters[-1] # parent branch diameter
        alpha,gamma = self.get_branching_parameters(branch,direction=direction)
        diameter_1 = d0 / np.power(1+np.power(alpha,gamma),1./gamma)
            
        diameter = alpha*diameter_1
        return [diameter_1,diameter]
        
    def calculate_branching_angles(self,branch,direction=None,gen=None):

        alpha,gamma = self.get_branching_parameters(branch,direction=direction)
        brad = np.arccos( (np.power((1+np.power(alpha,3)),4./3.) + np.power(alpha,4) - 1.) / (2.*np.square(alpha)*np.power(1.+np.power(alpha,3),2./3.)) )
        brad_1 = np.arccos( (np.power((1+np.power(alpha,3)),4./3.) + 1. - np.power(alpha,4)) / (2.*np.power(1.+np.power(alpha,3),2./3.)) )

        return np.rad2deg(brad_1), np.rad2deg(brad)
        
    def get_branching_parameters(self,branch,direction=None,gen=None):

        alpha = self.branching_alpha
        gamma = self.branching_gamma
            
        return alpha, gamma  
        
    def _initialiser_generator(self):
    
        return {    'world_direction':None,
                    'world_normal':None,
                    'branch_direction':None,
                    'branch_normal':None,
                    'length':0.,
                    'diameters':None,
                    'angles':None,
                    'branch_transverse_direction':None,
                    'interbranch_distance':None,
                    'last_branch_dist':None,
                    'label':'',
                    'branching_type':'t',
                    'stub_branch':False,
                    'creation_process':self,
                    'node':None,
                    'segment':None,
                    'end_point_coords':None,
                    'fully_defined':False,
                }  
                
    def get_branching_factor(self,branch):
        return self.branching_factor   
