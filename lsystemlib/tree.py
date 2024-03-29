import numpy as np
arr = np.asarray
#norm = np.linalg.norm
dot = np.dot
from retinasim import geometry
norm = geometry.norm
from stl import mesh
import time
from matplotlib import pyplot as plt
import copy

from retinasim.lsystemlib.utilities import *
from retinasim.lsystemlib.growth_process import GrowthProcess
from retinasim.lsystemlib.euler_bernoulli import euler_bernoulli
from pymira.spatialgraph import Editor

xaxis,yaxis,zaxis = arr([1.,0.,0.]), arr([0.,1.,0.]), arr([0.,0.,1.])

"""
Set of classes for defining spatial graphs
"""
                   
class TreeProperty(object):

    def __init__(self, name, clone=None, **kwargs):
    
        self.name = name
        self.verbose = 0
    
        for k,v, in zip(self.properties.keys(),self.properties.values()):
            if k in kwargs.keys():
                setattr(self,k,kwargs[k])
            else:
                setattr(self,k,v)
                
        if clone is not None:
            self = self.copy(clone) #copy.deepcopy(self)
             
        for k in kwargs.keys():
            if k not in self.properties.keys():
                if self.verbose>0:
                    print('Error: {} is not a valid keyword for a {} object!'.format(k,self.name))
                    
        self.creation_time = time.time() 
        self.update() 
        
    def set_property(self,prop,value):
        if prop in self.properties.keys():
            setattr(self,prop,value)
            self.update(changed_property=prop)
            
    def update(self,*args,**kwargs):
        pass
        
    def propogate_normal(self,dir1,dir2,norm1):
        # Calculate normal for segments with direction dir1 and dir2, with norm1 associated with dir1
        if np.all(dir1==dir2):
            return norm1
        
        if np.all(dir2==zaxis):
            norm2 = xaxis
        elif np.all(dir2==-zaxis):
            norm2 = -xaxis
        else:
            R = geometry.align_vectors(dir1,dir2)
            norm2 = dot(R,norm1)
        return norm2
        
    def __deepcopy__(self):
 
        new = copy.copy(self)
        for k in self.__dict__.keys():
            #print(k)
            srcval = getattr(self,k)
            if type(srcval) in [list,dict]:
                val = srcval.copy()
            elif isinstance(srcval,TreeProperty):
                val = srcval
            else:
                val = copy.deepcopy(srcval)
            setattr(new,k,val)
        return new
        
    def print(self):
        print(self.name)
        for k in self.properties.keys():
            print(k,getattr(self,k))

# Branches are collections of segments                
class Branch(TreeProperty):
    
    def __init__(self, **kwargs):
    
        name = 'branch'
        
        self.properties = {
                   'id':-1,
                   'order':-1,
                   'status':'',
                   'type':'',
                   'category':'',
                   'label':'',
                   'connected':False,
                   'segments':[],
                   'daughters':[],
                   'segment_count':0,
                   'tip_segment':None,
                   'root_segment':None,
                   #'daughter_order':[],
                   'daughter_count':0,
                   #'newest_daughter':None,
                   'cycle':-1,
                   'initial_direction_brf':None, # initial growth direction in parnet branch reference frame (brf)
                   'next_segment_params':None,
                   'growth_direction':None,
                   'growth_normal':None,
                   'next_diameter':None,
                   'interbranch_distance':None,
                   'parent':None,
                   'tree':None,
                   'start_cycle':0,
                   'terminal':None,
                   'fitness':0.,
                   'growth_process':None,
                   'creation_process':None,
                   'lock_paths':False,
                   }  
                   
        super().__init__(name,**kwargs)
        
        # Set value for inherited properties (i.e. properties inherited from parent branches)                 
        if self.tree is not None:
            if len(self.tree.inherited_branch_property)>0:
                for prop in self.tree.inherited_branch_property:
                    setattr(self,prop['name'],prop['init'])
        
        self.update()
        
    def set_growth_process(self, proc):
        if proc is None:
            proc = GrowthProcess()
        self.growth_process = proc
        
    def get_growth_process(self):
        if self.growth_process is None:
            self.set_growth_process(None)
        return self.growth_process
        
    def update(self,changed_property=None,**kwargs):
        if self.status=='':
            self.set_property('status','active')
            
        if len(self.daughters)>0:
            #This is a hack becuase parent reference fro branches seem to be inconsistently stored....
            for branch in self.daughters:
                if branch.status=='removed':
                    self.remove_daughter(branch)
        
        if self.parent is None:
            self.order = 0
        else:
            self.order = self.parent.order + 1
            if self.label is None or self.label=='':
                self.label = self.parent.label 
                
            if self.category=='':
                self.category = self.parent.category 
                
            if self.growth_process is None:
                self.growth_process = self.parent.growth_process
        
    def consistency_check(self, tree=None):
    
        if tree is None:
            tree = self.tree
    
        if self.root_segment not in tree.segments:
            err = 'Branch {} root segment ({}) not in tree list'.format(self.id,self.root_segment.id)
            
        cur_seg = self.root_segment
        visited = [cur_seg]
        while True:
            res,err = cur_seg.consistency_check()
            if err!='':
                err = 'Branch {}: {}'.format(self.id,err)
                return False, err
                
            next_seg = []
            for dseg in cur_seg.end_node.downstream_segments:
                if dseg.branch.id==self.id:
                    next_seg.append(dseg)
                    
            if len(next_seg)==0:
                break
            elif len(next_seg)==1:
                cur_seg = next_seg[0]
                if cur_seg in visited:
                    err = 'Branch {} has a loop at seg {}'.format(self.id,cur_seg.id)
                    return False, err
            else:
                err = 'Branch {} has more than one daughter segment at seg {}'.format(self.id,cur_seg.id)
                return False, err
                
        return True, err
        
    def associate_segment(self, segment, keep_tip=False, insert_index=None):

        if segment not in self.segments:  
            if insert_index is None:
                self.segments.append(segment)
            else:
                self.segments.insert(insert_index,segment)
                if insert_index==len(self.segments)-1:
                    keep_tip = False # Override
            self.segment_count += 1
            if len(self.segments)==1:
                self.root_segment = segment
            #if self.tip_segment is None or segment.parent==self.tip_segment:
            if not keep_tip:
                self.set_tip_segment(segment)
                self.set_growth_direction()
        
        if self.tree is not None:
            self.tree.associate_segment(segment)
        
    def remove_segment(self, segment, tree=True):

        if segment in self.segments:
            segment.status = 'removed'
            self.segments.remove(segment)
            
        if self.tip_segment==segment:
            if len(segment.start_node.upstream_segments)>0:
                if segment.start_node.upstream_segments[0].branch.id==segment.branch.id:
                    #print('Shifting tip segment upstream')
                    self.set_tip_segment(segment.start_node.upstream_segments[0])
                else:
                    # No segments left
                    self.set_tip_segment(None)
            #else:
            #    import pdb
            #    pdb.set_trace()
            else:
                print('Finding tip node')
                ns,ss = self.get_branch_nodes()
                self.set_tip_segment(ss[-1])
        
        if self.tree is not None and segment in self.tree.segments and tree is True:
            self.tree.remove_segment(segment)     
            
    def remove_all_segments(self, update_tree=True, update_nodes=True, update_daughters=True):
    
        if update_daughters:
            for daughter in self.daughters.copy():
                self.tree.remove_branch(daughter)
            
        for seg in self.segments:
            if update_tree and self.tree is not None:
                self.tree.remove_segment(seg)
            if update_nodes:
                seg.start_node.remove_segment(seg)
                seg.end_node.remove_segment(seg)
        self.segments = []
        self.segment_count = 0
        self.tip_segment = None
        self.root_segment = None
            
    def associate_daughter(self, daughter):
        if daughter in self.daughters:
            return
        self.daughters.append(daughter)
        #self.daughter_order.append(self.daughter_count)
        self.daughter_count += 1
        #self.newest_daughter = daughter
        
    def remove_daughter(self, daughter, update_tree=False):
        if daughter in self.daughters:
            self.daughters.remove(daughter)
            
        if self.tree is not None and update_tree is True:
            self.tree.remove_branch(daughter)
            
    def get_nodes(self):
        nodes = []
        for seg in self.segments:
            nodes.append(seg.start_node)
            nodes.append(seg.end_node)
        nodes = arr(list(set(nodes)))
        return nodes
        
    def euler(self):

        nodes, segments = self.get_branch_nodes()
        coords = arr([x.coords for x in nodes])
        radii = arr([x.diameters/2 for x in segments]).flatten()
        dz = euler_bernoulli(coords,r=radii)
        
        for i,node in enumerate(nodes):
            node.translate(arr([0.,0.,dz[i]]))
    
    def get_branch_nodes(self):
        current_seg = self.root_segment
        nodes = [current_seg.start_node,current_seg.end_node]
        segments = [current_seg]
        while True:
            next_seg = None
            for daughter in current_seg.end_node.downstream_segments:
                if daughter.branch==self:
                    next_seg = daughter
                    nodes.append(daughter.end_node)
                    segments.append(daughter)
            if next_seg is None:
                break
            else:
                current_seg = next_seg
        return nodes, segments
                    
    
    #def get_coords(self):
    #    current_segs = current_segment
    #    coords = [curseg.start_node.coords]
    #    while True:
     #       next_segs = [x for x in current_segment.downstream_segments]
    #        for seg in current_segs:
    #            if len(seg.downstream_segments)>0:
    #                next_segs.append(seg.downstream_segments)
        
    def get_branching_nodes(self):
        nodes = []
        for seg in self.segments:
            if seg.start_node.type=='branch':
                nodes.append(seg.start_node)
            if seg.start_node.type=='branch':
                nodes.append(seg.end_node)
        nodes = arr(list(set(nodes)))
        return nodes

            
    def get_segment_lengths(self):
        return arr([seg.length for seg in self.segments])
            
    def get_length(self):
        return np.sum(self.get_segment_lengths())
        
    def get_node_connecting_path(self, start_node=None,end_node=None,parents=False):
    
        # finds the path between two nodes that are directly upstream from each other
    
        node_found = False
        end_node_found = False
        
        default_null = []
        
        if end_node is None:
            end_node = self.tip_segment.end_node
        if start_node is None:
            start_node = self.root_segment.start_node
            
        # Check we are not looking at the same node
        if start_node==end_node:
            return default_null
            
        # Work backwards up the tree from the end node
        cur_seg = end_node.upstream_segments[0] # assumes only one upstream segment
        start_branch = cur_seg.branch
        path = [cur_seg]
        while True:
            if cur_seg.start_node==start_node:
                return path
                
            next_seg = cur_seg.start_node.upstream_segments
            if len(next_seg)==0:
               return default_null
               
            if not parents and next_seg[0].branch!=start_branch:
                return default_null
               
            path.append(next_seg[0])
            cur_seg = next_seg[0]
        
    def get_distance_between_nodes(self,start_node=None,end_node=None,return_path=False,parents=False):
    
        # Distance between nodes within the same branch
        
        if end_node is None:
            end_node = self.tip_segment.end_node
        if start_node is None:
            start_node = self.root_segment.start_node

        path = self.get_node_connecting_path(start_node=start_node,end_node=end_node,parents=parents)
        
        if len(path)==0:
            if return_path:
                return 0., []
            else:
                return 0.
            
        length = np.sum([x.get_length() for x in path])
        if return_path:
            return length, path
        else:
            return length
        
    def get_daughter_directions(self,process_name=None):
        """
        Returns daughter branch directions
        """
        if process_name is not None:
            return [x.get_growth_direction()['direction'] for x in self.daughters if x.status=='active' and x.creation_process.name==process_name]
        else:
            return [x.get_growth_direction()['direction'] for x in self.daughters if x.status=='active']
        
    def get_daughter_initialisation_branch_directions(self,process_name=None):
        """
        Returns daughter branch directions in branch reference frame
        """
        if process_name is not None:
            return [x.initial_direction_brf for x in self.daughters if x.status=='active' and x.creation_process.name==process_name]   
        else:
            return [x.initial_direction_brf for x in self.daughters if x.status=='active']   

    def get_daughter_directions_and_distances(self):
    
        seg = self.root_segment
        length = seg.length
        res = []
        while True:
            next_seg = None
            daughters = seg.get_daughters()
            if daughters is not None:
                for daughter in daughters:
                    if daughter.branch!=self:
                        if len(seg.get_daughters())>0:
                            res.append({'seg':daughter,'direction':daughter.branch.initial_direction_brf,'length':length,'creation_process':daughter.creation_process.name})
                    else:
                        next_seg = daughter
                        length += daughter.length
                        
                if next_seg is None:
                    break
                else:
                    seg = next_seg
            else:
                import pdb
                pdb.set_trace()
        return res
        
    def get_growth_direction(self):
        gd,gn = self.set_growth_direction()
        return {'direction':self.growth_direction, 'normal':self.growth_normal}
            
    def set_growth_direction(self):
        if self.tip_segment is None:
            return None, None
        pdirection = self.tip_segment.direction
        pnormal = self.tip_segment.normal
        self.growth_direction = pdirection / norm(pdirection)
        self.growth_normal = pnormal / norm(pnormal)
        return self.growth_direction, self.growth_normal  
            
    def get_growth_normal(self):
        if self.growth_normal is not None:
            return self.growth_normal
        else:
            segment = self.get_tip_segment()
            return segment.normal #s[-1]
            
    #def get_branches_within_distance_from_point(self,point
        
    def get_distance_from_last_branch(self,end_node=None, process_name=None,return_node=False,parents=False):
        if len(self.daughters)==0:
            if return_node:
                return None,None
            else:
                return None

        dists,nodes = [],[]
        cur_branch = self
        count = 0
        while True:
            # Loop through each of the branch's daughters and calculate their distance from the tip (or supplied end node)
            for daughter in cur_branch.daughters:
                if process_name is None or daughter.creation_process.name==process_name:
                    branch_node = daughter.root_segment.start_node #  self.daughters[-1].root_segment.start_node
                    dist = cur_branch.get_distance_between_nodes(start_node=branch_node,end_node=end_node,parents=True)
                    if dist is not None:
                        dists.append(dist)
                        nodes.append(branch_node)
                        
            if len(dists)==0 and parents==True:
                cur_branch = cur_branch.parent
                if cur_branch is None:
                    break
            else:
                break
            count += 1
            if count>100:
                import pdb; pdb.set_trace()
                
        if len(dists)>0:
            if return_node:
                mn = np.argmin(dists)
                return dists[mn],nodes[mn]
            else:
                return np.min(dists) 
        else:
            if return_node:
                return None,None
            else:
                return None
                
    def get_distance_to_next_branch(self,start_node=None, process_name=None,return_node=False,parents=False):
        if len(self.daughters)==0:
            if return_node:
                return None,None
            else:
                return None

        dists,nodes = [],[]
        cur_branch = self
        count = 0
        while True:
            # Loop through each of the branch's daughters and calculate their distance from the tip (or supplied end node)
            for daughter in cur_branch.daughters:
                if process_name is None or daughter.creation_process.name==process_name:
                    branch_node = daughter.root_segment.start_node #  self.daughters[-1].root_segment.start_node
                    dist = cur_branch.get_distance_between_nodes(start_node=start_node,end_node=branch_node,parents=True)
                    if dist is not None:
                        dists.append(dist)
                        nodes.append(branch_node)
                        
            break
            count += 1
            if count>100:
                import pdb; pdb.set_trace()
                
        if len(dists)>0:
            if return_node:
                mn = np.argmin(dists)
                return dists[mn],nodes[mn]
            else:
                return np.min(dists) 
        else:
            if return_node:
                return None,None
            else:
                return None       
        
    def get_tip_node(self):
        if self.tip_segment is None:
            return None
        return self.tip_segment.end_node
        
    def set_tip_segment(self, segment):
        if segment is None:
            self.tip_segment = segment
        elif segment in self.segments and len([x for x in segment.end_node.downstream_segments if x.id==self.id])==0:
            self.tip_segment = segment
            self.set_growth_direction()
        else:
            if len(self.segments)==0 or self.status!='active':
                return
            #import pdb
            #pdb.set_trace()
            
    def set_root_segment(self, segment):
        if segment is None:
            self.root_segment = segment
        elif segment in self.segments and len([x for x in segment.end_node.downstream_segments if x.id==self.id])==0:
            self.root_segment = segment
        else:
            if len(self.segments)==0 or self.status!='active':
                return
        
    def get_tip_direction(self):
        sdirection = self.tip_segment.directions[-1]
        snormal = self.tip_segment.normal #s[-1]
        return sdirection,snormal  
        
    def rotate_branch(self, theta, rot_axis):
        """
        Rotate branch about start node
        Theta in *degrees*
        """
        n0 = self.root_segment.start_node # Rotation point at branch start node
        ds_nodes,ds_segs = n0.get_downstream_nodes()
        for cnode in ds_nodes:
            coords_p = geometry.rotate_about_axis(cnode.coords,rot_axis,theta,centre=n0.coords)
            cnode.set_property('coords',coords_p)
        for cseg in ds_segs:
            dir_p = geometry.rotate_about_axis(cseg.direction,rot_axis,theta,centre=n0.coords)
            cseg.set_property('direction',dir_p)
            normal_p = geometry.rotate_about_axis(cseg.normal,rot_axis,theta,centre=n0.coords)
            cseg.set_property('normal',normal_p)
       
    def print(self):
        print('Branch {}'.format(self.id))
        print('Label: {}'.format(self.label))
        print('Category: {}'.format(self.category))
        print('Type: {}'.format(self.type))
        print('Status: {}'.format(self.status))
        print('Length: {:.2f} um'.format(self.get_length()))
        if self.interbranch_distance is not None:
            print('Interbranch distance: {:.2f} um'.format(self.interbranch_distance))
        print('Number of segments: {}'.format(len(self.segments)))
        print('Number of daughters: {}'.format(len(self.daughters)))

class Node(TreeProperty):

    """ 
    Nodes define the start point, or branching points in a tree, and are connected together via segments
    type: 0=root, 1=inline connection, 2=branching node, 3=tip
    status: 0=unassigned, 1=active, 2=inactive
    """
    
    def __init__(self, **kwargs):
    
        name = 'node'
        self.properties = {
                   'coords':None,
                   'id':-1,
                   'status':'',
                   'type':'',
                   'category':'',
                   'upstream_segment_count':0,
                   'downstream_segment_count':0,
                   'segment_count':0,
                   'upstream_segments':[],
                   'downstream_segments':[],
                   'cycle':-1,
                   'density':0.,
                   'fitness':0.,
                   }
        super().__init__(name,**kwargs)
        
        #if self.tree is not None:
        #    if len(self.tree.inherited_node_property)>0:
        #        for prop in self.tree.inherited_node_property:
        #            setattr(self,prop['name'],prop['init'])
        
        self.update()
        
    def associate_upstream_segment(self, segment):
        if segment not in self.upstream_segments:
            self.upstream_segments.append(segment)
            self.set_property('upstream_segment_count',self.upstream_segment_count+1)
            
    def associate_downstream_segment(self, segment):
        if segment not in self.downstream_segments:
            self.downstream_segments.append(segment)
            self.set_property('downstream_segment_count',self.downstream_segment_count+1)  
            
    def set_upstream_segments(self, segments):
        self.upstream_segments = segments
        self.upstream_segment_count = len(segments)
        self.set_property('upstream_segment_count',self.upstream_segment_count)               
            
    def set_downstream_segments(self, segments):
        self.downstream_segments = segments
        self.downstream_segment_count = len(segments)
        self.set_property('downstream_segment_count',self.downstream_segment_count) 

    def remove_segment(self, segment):
        if segment in self.upstream_segments:
            self.upstream_segments.remove(segment)
            self.set_property('upstream_segment_count',self.upstream_segment_count-1)
        if segment in self.downstream_segments:
            self.downstream_segments.remove(segment)
            self.set_property('downstream_segment_count',self.downstream_segment_count-1)
        
    def remove_all_segments(self):
        self.upstream_segments = []
        self.set_property('upstream_segment_count',0)
        self.downstream_segments = []
        self.set_property('downstream_segment_count',0)
        
    def update(self,changed_property=None,**kwargs):
    
        if self.coords is None:
            if self.verbose>0:
                print('No coordinates defined for node {}'.format(self.id))    
        
        if self.status=='':
            self.status = 'active'
            
        if len(self.upstream_segments)+len(self.downstream_segments)==0:
            self.type = 'orphan'
        elif len(self.upstream_segments)==1 and len(self.downstream_segments)==0:
            self.type = 'end'
        elif len(self.upstream_segments)==0 and len(self.downstream_segments)==1:
            self.type = 'root'
        elif len(self.upstream_segments)==1 and len(self.downstream_segments)==1:
            self.type = 'inline'
        else:
            self.type = 'branch'  
            
        if changed_property=='coords':
            [x.update() for x in self.upstream_segments]
            [x.update() for x in self.downstream_segments]
            
    def translate(self,dx,propagate=False):
    
        self.set_property('coords',self.coords+dx)
        print('Translating node: {}'.format(self.id))
        
        if propagate:
            ds_nodes,_ = self.get_downstream_nodes()
            if self in ds_nodes:
                ds_nodes.remove(self)
            ds_nodes = list(set(ds_nodes)) # Remove any repeats
            for cnode in ds_nodes:
                cnode.translate(dx,propagate=False)
                        
    def get_downstream_nodes(self, _result=None, _segments=None):
    
        if _result is None:
            _result = [self]
        elif self not in _result:
            _result.append(self)
            
        if _segments is None:
            _segments = []
        
        for seg in self.downstream_segments:
            if seg not in _segments:
                _segments.append(seg)
                seg.end_node.get_downstream_nodes(_result=_result,_segments=_segments)
            
        return _result,_segments
        
    def get_upstream_nodes(self, result=None):
    
        if result is None:
            result = [self]
        elif self not in result:
            result.append(self)
        
        for seg in self.upstream_segments:
            seg.start_node.get_upstream_nodes(result=result)
            
        return result        
        
# Segments connect nodes together, via edges
        
class Segment(TreeProperty):
    
    def __init__(self, **kwargs):
    
        name = 'segment'
        self.properties = {
                   'coords':None,
                   'id':-1,
                   'status':'incomplete',
                   'category':'',
                   'start_node':None,
                   'end_node':None,
                   'direction':None,
                   'normal':None,
                   'initial_direction':None,
                   'initial_normal':None,
                   'directions':None,
                   'normals':None,
                   'branch':None,
                   'diameter':0.,
                   'diameters':None,
                   'lengths':None,
                   'status':'active',
                   'type':'',
                   'cycle':-1,
                   'points':None,
                   'npoints':-1,
                   'parent':None,
                   #'daughters':[],
                   #'daughter_count':0,
                   #'newest_daughter':None,
                   'use_parent':True,
                   'create_nodes':False,
                   # CCO fields
                   'Leff':None,
                   'Rstar':None,
                   'flows':[],
                   'flow':None,
                   'branching_fraction':None,
                   }  
    
        super().__init__(name,**kwargs)
        
        if self.branch is not None and self.branch.tree is not None:
            if len(self.branch.tree.inherited_segment_property)>0:
                for prop in self.branch.tree.inherited_segment_property:
                    setattr(self,prop['name'],prop['init'])
        
        self.update()
        
        if self.verbose>1:
            print('Segment created {}'.format(self.id))
        
    def update(self,changed_property=None,**kwargs):
            
        if self.diameter<=0.:
            self.diameter = 10. # um
            
        if self.start_node is not None:
            if self not in self.start_node.downstream_segments:
                self.start_node.associate_downstream_segment(self)
        if self.end_node is not None:
            if self not in self.end_node.upstream_segments:
                self.end_node.associate_upstream_segment(self)
            
        if self.end_node is not None and self.start_node is not None:
            direction = self.end_node.coords - self.start_node.coords
            direction = direction / norm(direction)
            self.direction = direction
            #print('Set segment {} direction: {}'.format(self.id,self.direction))
            
        if self.direction is not None and self.normal is None:
            if np.all(self.direction==zaxis):
                self.normal = xaxis
            elif np.all(self.direction==-zaxis):
                self.normal = -xaxis
            else:
                self.normal = np.cross(zaxis,self.direction) 
            
        # Try to set the initial direction again, if a parent wasn't passed
        if self.initial_direction is None:
            if self.parent is not None and self.parent.direction is not None:
                self.initial_direction = self.parent.direction.copy()
                self.initial_normal = self.parent.normal.copy()
            elif self.direction is not None:
                self.initial_direction = self.direction.copy()
                self.initial_normal = self.normal.copy()
            
        if self.start_node is not None and self.end_node is not None:
            if self.coords is None:
                self.create_default_edge()            
            elif np.any(self.coords[0,:]!=self.start_node.coords) or np.any(self.coords[-1,:]!=self.end_node.coords):
                self.create_default_edge()
    
        # Check consistency
        if self.coords is not None and (self.directions is None or self.normals is None or changed_property=='coords'):
        
            self.directions, self.normals = [],[]
            
            prev_norm = self.initial_normal
            prev_dir = self.initial_direction

            for i,coord in enumerate(self.coords):
                if i>0:
                    direction = self.coords[i] - self.coords[i-1]
                    direction = direction / norm(direction)
                    self.directions.append(direction)
                    
                    normal = self.propogate_normal(prev_dir.astype('float'),direction.astype('float'),prev_norm.astype('float'))
                    prev_norm = normal
                    prev_dir = direction
                    
                    self.normals.append(normal)
                
            self.directions = np.asarray(self.directions)
            self.normals = np.asarray(self.normals)
            #print('Set edge {} directions: {}'.format(self.id,self.directions))
            
        if self.directions is not None and self.normals is None:
            if np.all(self.direction==zaxis):
                self.normal = xaxis
            elif np.all(self.direction==-zaxis):
                self.normal = -xaxis                
            else:
                self.normal = np.cross(zaxis,self.direction)
                
        self.length = self.get_length()
        
        #if self.status!='active' and self.start_node is not None and self.end_node is not None:
        #    self.set_property('status','active')
 
    def consistency_check(self, tree=None):
        err = ''
        if tree is None:
            tree = self.branch.tree
        
        # Seg - node
        if self.start_node is None:
            err = 'Seg {} start node is None'.format(self.id)
        elif np.any(np.isnan(self.start_node.coords)):
            err = 'Seg {} start node contains NaN'.format(self.start_node.id)
        elif self.end_node is None:
            err = 'Seg {} end node is None'.format(self.id)
        elif np.any(np.isnan(self.end_node.coords)):
            err = 'Seg {} end node contains NaN'.format(self.end_node.id)
        elif self.start_node not in tree.nodes:
            err = 'Seg {} start node ({}) not in tree nodes'.format(self.id,self.start_node.id)
        elif self.end_node not in tree.nodes:
            err = 'Seg {} end node ({}) not in tree nodes'.format(self.id,self.end_node.id)
        elif self not in self.start_node.downstream_segments:
            err = 'Seg {} not in start node downstream list'.format(self.id)
        elif self not in self.end_node.upstream_segments:
            err = 'Seg {} not in end node upstream list'.format(self.id)
            
        if self.parent is not None:
            if self.parent not in self.start_node.upstream_segments:
                err = 'Seg {} parent ({}) is not in start node ({}) upstream list'.format(self.id,self.parent.id,self.start_node.id)
                
        if np.any(np.isnan(self.coords)):
            err = 'Seg {} contians NaN'.format(self.id)
            
        # Seg - branch
        if self.branch is None:
            err = 'Seg {} does not belong to a branch'.format(self.id)
        elif self not in self.branch.segments:
            err = 'Seg {} is not in branch {} segment list'.format(self.id,self.branch.id)            
        
        if err!='':
            return False, err
        return True, ''
        
    def create_default_edge(self):
        npoints = 2
        points = np.linspace(self.start_node.coords,self.end_node.coords,npoints)
                
        if self.diameter>0.:
            self.diameters = np.zeros(npoints) + self.diameter
        else:
            self.diameters = np.zeros(npoints) + 1.
            
        self.lengths = arr([norm(points[i]-points[i-1]) for i in range(1,npoints)])
        self.flows = np.zeros(npoints) - 1.
        self.coords = points
        
    def set_edge_points(self,coords,diameters=None):
        if np.any(coords[0,:]!=self.start_node.coords):
            return False
        if np.any(coords[-1,:]!=self.end_node.coords):
            return False
            
        self.directions,self.normals = None,None
        
        if diameters is not None and diameters.shape[0]==coords.shape[0]:
            self.diameters = diameters
        elif self.diameter>0.:
            self.diameters = np.zeros(coords.shape[0]) + self.diameter
        else:
            self.diameters = np.zeros(coords.shape[0]) + 1.
            
        npoints = coords.shape[0]
        self.lengths = arr([norm(coords[i]-coords[i-1]) for i in range(1,coords.shape[0])])
        
        self.set_property('coords',coords)
        return True
                
    def get_length(self):
        if self.coords is None:
            return 0.
        length = np.sum([norm(self.coords[i]-self.coords[i-1]) for i,x in enumerate(self.coords[1:])])
        return length   
    
    def get_direction(self):    
        return {'direction':self.direction, 'normal':self.normal}         
            
    def get_start_coords(self):
        if self.start_node is None:
            return None
        return self.start_node.coords
        
    def get_end_coords(self):
        if self.start_node is None:
            return None
        return self.start_node.coords
        
    def get_edge_coords(self, fix_if_broken=True):
        if np.any(self.start_node.coords!=self.coords[0,:]) or np.any(self.end_node.coords!=self.coords[-1,:]):
            if fix_if_broken:
                self.create_default_edge()
            else:
                return None
        return self.coords
        
    def get_edge_normals(self):
        return self.normals
        
    def associate_branch(self, branch):
        self.set_property('branch',branch)
        
    #def associate_daughter(self, daughter):
    #    if daughter in self.daughters:
    #        return
    #    self.daughters.append(daughter)
    #    self.daughter_count += 1
    #    self.newest_daughter = daughter
    
    def get_daughters(self):
        if self.end_node is None:
            return None
        return self.end_node.downstream_segments
        
    def divide(self, coord=None):
        
        """
        From: nO----------n1 (seg0)
        to:   nO----n2----n1 (seg0, seg1 [new])
        
        """
        
        if coord is None:
            coords = np.linspace(self.start_node.coords,self.end_node.coords,3)
            coord = coords[1,:]
                               
        seg1 = self.__deepcopy__() #copy.deepcopy(self)
        seg0 = self
        n0 = seg0.start_node
        n1 = seg0.end_node
        n2 = Node(coords=coord) # n2 node
        
        uss = seg0.start_node.upstream_segments.copy()
        dss0 = seg0.start_node.downstream_segments.copy()
        dss1 = seg0.end_node.downstream_segments.copy()

        # Segment properties
        seg0.set_property('end_node',n2)
        seg1.set_property('start_node',n2)
        # Branch properties
        insert_index = seg0.branch.segments.index(seg0) + 1
        seg0.branch.associate_segment(seg1,keep_tip=True,insert_index=insert_index)
        seg0.branch.tree.associate_node(n2)
        
        # Gets taken care of in associate_segment function? Or somewhere else?
        #if branch.tip_segment==seg0:
        #    branch.tip_segment = seg1
        
        seg1.parent = seg0
        
        # Connections
        # N0 connections should not change
        #n0.set_upstream_segments(uss)
        #n0.set_downstream_segments(dss0)
        
        # New middle node
        n2.set_upstream_segments([seg0])
        n2.set_downstream_segments([seg1])
        
        # End node
        n1.set_upstream_segments([seg1])
        #n1.set_downstream_segments(dss1)

        # Re-parent any existing donwstream segments        
        if len(dss1)>0:
            for ds_seg in dss1:
                ds_seg.parent = seg1
        
        #print('**Dividing segment {} in two ({},{})'.format(self.id,seg0.id,seg1.id))
        print([x.id for x in uss],[x.id for x in dss0],[x.id for x in dss1])
        #print('**uss_n0:{},dss_n0:{},uss_n2:{},dss_n2:{},uss_n1:{},dss_n1:{}'.format([x.id for x in n0.upstream_segments],[x.id for x in n0.downstream_segments],[x.id for x in n2.upstream_segments],[x.id for x in n2.downstream_segments],[x.id for x in n1.upstream_segments],[x.id for x in n1.downstream_segments]))
        
        if seg0.length==0. or seg1.length==0.:
            breakpoint()
        
        return seg1
        
    def get_downstream_segments(self, result=None):
    
        if result is None:
            result = [self]
        elif self not in result:
            result.append(self)
        
        for seg in self.get_daughters():
            seg.get_downstream_segments(result=result)
            
        return result

class Tree(object):

    def __init__(self,prefix='',path=None,store_graph=False,store_mesh=False,store_every=1,domain=None,ignore_domain=False,
                      write_longitudinal=True,verbose=2,params=None):

        self.verbose = verbose 
        self.inherited_branch_property = []
        self.inherited_segment_property = []
        self.inherited_node_property = []
        self.allocated = []
        
        #self.add_inherited_property(name='growth_process',dtype='object',type='branch',init=GrowthProcess())
        
        self.unit_length = 10. # um
        self.unit_diameter = 10. # um

        self.nodes = []
        self.branches = []
        self.segments = []
        
        self.n_edge_points = 2
        
        self.nodeCount = 0
        self.edgeCount = 0
        self.segmentCount = 0
        self.branchCount = 0
        
        self.root_segment = None
        self.root_node = None
        
        self.inlet_node_id = -1
        self.outlet_node_id = -1

        # Which fields to write into graph files
        self.graph_params = [ 
                      {'name':'Radius','element_type':'node','data_field':'radii'},
                      #{'name':'Cycle','element_type':'node','data_field':'cycle'},
                      #{'name':'Order','element_type':'branch','data_field':'order'},
                      {'name':'Category','element_type':'branch','data_field':'category'},
                    ]
            
    def print(self):
        print('Tree')
        print('# branches: {}'.format(len(self.branches)))
        print('# segments: {}'.format(len(self.segments)))
        print('# nodes: {}'.format(len(self.nodes)))        
                    
    def add_inherited_property(self,name='',dtype='int',type='branch',init=-1):
        if type=='branch':
            self.inherited_branch_property.append({'name':name,'dtype':dtype,'init':init})
        elif type=='segment':
            self.inherited_segment_property.append({'name':name,'dtype':dtype,'init':init})
        elif type=='node':
            self.inherited_node_property.append({'name':name,'dtype':dtype,'init':init})            
            
        #TODO: Add support for inherited node and segment properties
            
    def get_segment_by_id(self,id):
        return [x for x in self.segments if x.id==id] 
        
    def get_node_by_id(self,id):
        return [x for x in self.nodes if x.id==id] 
        
    def get_branch_by_id(self,id):
        return [x for x in self.branches if x.id==id] 

    def consistency_check(self):
    
        err = ''
        for seg in self.segments:
            res,err = seg.consistency_check()
            if err!='':
                return False, err
                
        for branch in self.branches:
            res,err = branch.consistency_check()

        return True, ''    
        
    def segment_collision_test_2d(self, seg1, seg2, padding=0.):

        v1dir = (seg1.end_node.coords-seg1.start_node.coords)
        v1len = np.linalg.norm(v1dir)
        v1dir = v1dir / v1len
        v2dir = (seg2.end_node.coords-seg2.start_node.coords)
        v2len = np.linalg.norm(v2dir)
        v2dir = v2dir / v2len
        
        if v1len==0 or v2len==0:
            print('Warning: segment_collision_test_2d: zero length vector encountered!')
            return False,None
        
        collision = geometry.segments_intersect_2d( seg1.start_node.coords-v1dir*padding,
                                                    seg1.end_node.coords+v1dir*padding,
                                                    seg2.start_node.coords-v2dir*padding,
                                                    seg2.end_node.coords+v2dir*padding)
        return collision # tuple - collision flag and intersection point
        
    def segment_collision_check(self,segment1,same_kind=False,dimensions=2):
    
        # Check for collisions with a supplied segment object
        rad1 = segment1.diameters[-1]/2.

        realign = []
        # Check every segment for collisions with current segment
        for segment2 in self.segments:

            # Get target segment start/end node coordinates and radii
            rad2 = segment2.diameters[-1]/2.
            
            # Collision detection
            if segment2.branch.category==segment1.branch.category:
                padding = np.max([rad1,rad2]) * 10.
            else:
                padding = 10. # um
            
            if dimensions==2:
                coll, intersection = self.segment_collision_test_2d(segment1, segment2, padding=padding)
            elif dimensions==3:
                # TODO: Add in 3D support
                print('Warning 3D vessel collisions not currently supported!')
                return None
            
            if coll:
                # If the collision is between vessels with the same category (e.g. artery-artery or vein-vein)
                # return a collision fault if they don't share a junction (i.e. are joined rather than colliding)
                if segment2.branch.category==segment1.branch.category:
                    if not(segment1.parent==segment2 or segment2.parent==segment1 or segment1.start_node==segment2.start_node or segment1.end_node==segment2.end_node):
                        return True, segment2, intersection

                # If the collision is between two different category segments
                elif same_kind is False: # Detect vein-artery collisions
                    return True, segment2, intersection
                        
        return False, None, None    # flag, collision segment, collision point   
        
    def collision_check(self,same_kind=True):
        
        # Performs collision detection across the whole network
        res = []
        for seg1 in self.segments:
            coll,seg2,intersection = self.segment_collision_check(seg1,same_kind=same_kind)
            if coll:
                res.append([seg1,seg2,intersection])
        return res
        
    # Get / set 
    
    def get_volume(self):
        return np.sum([x.get_length()*np.sum((x.diameters[1:]/2.)**2) for x in self.segments])*np.pi
        
    def get_extent(self):
        coords = arr([x.coords for x in self.nodes])
        if len(coords)>0:
            return coords.min(axis=0), coords.max(axis=0)
        else:
            return None, None
    
    def get_active_branches(self):
        return [x for x in self.branches if x.status=='active']
        
    def associate_branch(self,branch):
        if branch not in self.branches:
            self.branches.append(branch)
            #if branch.id<0:
            branch.id = self.branchCount
            self.branchCount += 1
        
    def add_branch(self,branch,**kwargs):
        
        kwargs['tree'] = self
        
        if branch is None:
            branch = Branch(**kwargs) 
 
        self.associate_branch(branch)
            
        if self.verbose>0:
            print('Added branch {} (creation proc: {})'.format(branch.id,branch.creation_process))
        
        return branch 
        
    def remove_branch_instance(self,branch):
        branch.status = 'removed'
        if branch in self.branches:
            self.branches.remove(branch)
            self.branchCount = len(self.branches)
        
    def remove_branch(self,branch,daughters=True):

        #if branch in self.branches:
        #if True:# self.verbose>0:
        #    print('Removing branch {}'.format(branch.id))
            
        branch.status = 'inactive'
            
        [self.remove_segment(x) for x in branch.segments]
        branch.remove_all_segments(update_daughters=True,update_tree=False,update_nodes=True)
        self.remove_branch_instance(branch)
        
        if branch.parent is not None:
            branch.parent.remove_daughter(branch, update_tree=False)
                
        if daughters:
            if len(branch.daughters)>0:
                for daughter in branch.daughters:
                    self.remove_branch(daughter,daughters=daughters) 
                    
    def propagate_property(self, branch, depth=0, **kwargs):
    
        for k,v in zip(kwargs.keys(),kwargs.values()):
            if k in self.__dict__.keys():
                setattr(self,k,v)
    
        if len(branch.daughters)>0:
            for daughter in branch.daughters:
                self.propagate_property(daughter,depth=depth+1,**kwargs)
        
    def create_branch(self,parent_branch=None,start_coord=None,end_coord=None,diameter=None,direction=None,length=None,no_segment=False,stub=False,inside_domain_function=None,**kwargs):
        """
        Helper function to create either an isolated (initial) branch with a single segment
        or a branch at the tip of an existing branch
        """
        
        if start_coord is None:
            if 'start_node' in kwargs.keys() and kwargs['start_node'] is not None:
                start_coord = kwargs['start_node'].coords
            else:
                if parent_branch is None:
                    return None
                else:
                    start_coord = parent_branch.tip_segment.end_node.coords
        
        kwargs['tree'] = self
        branch = self.add_branch(None,parent=parent_branch,**kwargs)
        
        # Create end cooordinate if the direction is provided
        if end_coord is None and direction is not None:
            if length is None:
                length = self.unit_length
            end_coord = start_coord + direction*length

        parent = None
        inherits = []
        if parent_branch is not None:
            if 'gen' in kwargs.keys() and kwargs['gen'] is not None and 'segment' in kwargs['gen'] and kwargs['gen']['segment'] is not None:
                parent = kwargs['gen']['segment']
            else:
                parent = parent_branch.tip_segment
            
            if len(self.inherited_branch_property)>0:
                for prop in self.inherited_branch_property:
                    try:
                        propname = prop['name']
                        propval = getattr(parent_branch,propname)
                        inherits.append({'name':propname,'value':propval})
                    except Exception as e:
                        print(e)

        if not no_segment:
            if np.all(start_coord==end_coord):
                breakpoint()
            segment,code = self.add_segment_to_branch(branch,parent=parent,start_coord=start_coord,end_coord=end_coord,diameter=diameter,inside_domain_function=inside_domain_function,**kwargs)
            if segment is None:
                self.remove_branch(branch)
                return None
            if stub:
                branch.status = 'inactive'
        
        if parent_branch is not None:   
            parent_branch.associate_daughter(branch)
            segment.start_node.type = 'branch'
            
        if len(inherits)>0:
            for e in inherits:
                setattr(branch,e['name'],e['value'])
            
        return branch
        
    def create_branch_at_node(self, node, parent_branch=None, end_coord=None, direction=None, normal=None, length=None, diameter=None, no_segment=False, inside_domain_function=None, **kwargs):
    
        kwargs['tree'] = self
        branch = self.add_branch(None,parent=parent_branch,**kwargs)
        
        start_coord = node.coords
        
        # Create end cooordinate if the direction is provided
        if end_coord is None and direction is not None:
            if length is None:
                length = self.unit_length
            end_coord = start_coord + direction*length

        parent = [x for x in node.upstream_segments if x.end_node==node][0]
        if not no_segment:
            segment,code = self.add_segment_to_branch(branch,normal=normal,start_node=node,parent=parent,start_coord=start_coord,end_coord=end_coord,diameter=diameter,inside_domain_function=inside_domain_function,**kwargs)
            if segment is None:
                self.remove_branch(branch)
                return None
        
        if parent_branch is not None:   
            parent_branch.associate_daughter(branch)
            if segment is not None:
                segment.start_node.type = 'branch'          
            
        return branch
        
    def associate_node(self, node):
    
        if node not in self.nodes:
            self.nodes.append(node)
            #if node.id<0:
            node.id = self.nodeCount
            self.nodeCount += 1
                    
    def add_node(self,node=None,ignore_domain=False,**kwargs):
    
        if node is None:
            node = Node(**kwargs)
            
        if node in self.nodes:
            return None, 'already added'

        #if not self.inside_domain(node.coords) and not ignore_domain:
        #    return None, 'outside_domain'
            
        self.associate_node(node)
        
        if self.verbose>1:
            print('Added node {}, {}'.format(node.id,node.coords))

        return node, ''
        
    def remove_node(self, node):
        if node in self.nodes:
            #print('Removing node:{}'.format(node.id))
            for seg in self.segments:
                if seg.start_node==node or seg.end_node==node:
                    print('Cannot remove node {}, still in segment {}'.format(node.id,seg.id))
                    return
            self.nodes.remove(node)
            self.nodeCount -= 1  
            
    def remove_orphan_nodes(self):
        for node in self.nodes:
            if len(node.upstream_segments)==0 and len(node.downstream_segments)==0:
                self.remove_node(node)
                
        
    def get_node_coords(self,active=True):
        if active:
            return arr([x.coords for x in self.nodes if x.status=='active'])
        else:
            return arr([x.coords for x in self.nodes])
            
    def get_edge_points(self,distance_coord=None):
        
        segments,points,n,categories,types,diameters = [],[],[],[],[],[]
        distances = []
        for seg in self.segments:
            if seg.coords is not None:
                points.extend(seg.coords)
                npts = seg.coords.shape[0]
            else:
                points.extend([seg.start_node.coords,seg.ed_node.coords])
                npts = seg.coords.shape[0]
            n.append(npts)
            categories.extend([seg.category]*npts)
            segments.extend([seg]*npts)
            types.extend([seg.type]*npts)
            diameters.extend(seg.diameters)
            
            if distance_coord is not None:
                distances.extend([norm(n-distance_coord) for n in seg.coords])
            else:
                distances = None
        return segments, arr(points), arr(n), categories, types, arr(diameters), arr(distances)
    
    def get_distance_from_edge_points(self, coord):    
        return arr([norm(n-coord) for n in points])
        
    def associate_segment(self, segment):
    
        if segment not in self.segments:
                
            if len(self.segments)==0:
                self.root_segment = segment
                self.root_node = segment.start_node
        
            self.segments.append(segment)
            segment.id = self.segmentCount
            self.segmentCount += 1
        
    def add_segment(self,segment,**kwargs):
    
        if self.verbose>1:
            branch = segment.branch
            if branch is not None:
                bid = branch.id
            else:
                bid = -1
            print('Added segment {} to branch {}, length: {} um'.format(segment.id,bid,segment.length))
    
        if segment is None:
            segment = self.create_segment(**kwargs)
            if segent is None:
                return None, 'invalid'
        if segment in self.segments:
            return None, 'already added'       
            
        self.associate_segment(segment)
        
        if self.verbose>1:
            print('Added segment {}'.format(segment.id))

        return segment,'' 
        
    def remove_segment_instance(self, segment):
        segment.status = 'removed'
        segment.parent = None 
        if segment in self.segments:
            self.segments.remove(segment)
            self.segmentCount -= 1
            
    def remove_segment(self, segment, propogate=True, depth=0, visited=None):
        #print('Removing segment {} (start node:{}, end node:{}, branch:{}) (depth:{})'.format(segment.id,segment.start_node.id,segment.end_node.id,segment.branch.id,depth))
        if visited is None:
            visited = [self]

        if propogate:

            daughters = segment.get_daughters().copy()
            for daughter in daughters: #end_node.downstream_segments:
                if daughter.branch!=segment.branch:
                    self.remove_branch(daughter.branch)
                elif daughter not in visited:
                    self.remove_segment(daughter, depth=depth+1, propogate=True, visited=visited)
                
        # Node associations with segment
        
        segment.start_node.remove_segment(segment)
        #if len(segment.end_node.downstream_segments)>0:
        #    for seg in segment.end_node.downstream_segments:
        #        self.remove_segment(seg, visited=visited, propogate=propogate)
        if segment.parent is not None:
            segment.parent.end_node.remove_segment(segment)
        
        # Branch associations
        if segment.branch is not None:
            segment.branch.remove_segment(segment, tree=False)
            # Update tip segment
            #if segment==segment.branch.tip_segment:
            #    if segment.parent is not None and self.parent.branch.id==segment.branch.id:
            #        segment.branch.set_tip_segment(segment)
            # If there are no more segments in the branch
            if len(segment.branch.segments)==0:
                self.remove_branch_instance(segment.branch)

        # Tree associations
        self.remove_segment_instance(segment)
        
        # Delete segment end node if orphaned
        if len(segment.end_node.downstream_segments)==0 and len(segment.end_node.upstream_segments)==0:
            self.remove_node(segment.end_node)            
            
        segment.status = 'removed'  
        
    def create_segment(self,**kwargs):
        segment = Segment(**kwargs)
        return segment, ''
        
    def add_segment_to_branch(self,branch,start_node=None,parent=None,start_coord=None,end_coord=None,direction=None,length=None,diameter=None,ignore_domain=False,inside_domain_function=None,**kwargs):

        if start_node is not None and parent is not None:
            #parent = start_node.segment
            pnormal = parent.normal.copy()
            pdirection = parent.direction.copy()
            category = parent.category
            start_coord = start_node.coords.copy()
        elif len(branch.segments)>0 or (branch.parent is not None and len(branch.parent.segments)>0):
            if len(branch.segments)==0:
                parent = branch.parent.tip_segment
            else:
                parent = branch.tip_segment #segments[-1]
            pnormal = parent.normal.copy()
            pdirection = parent.direction.copy()
            category = parent.category
            start_node = parent.end_node
            start_coord = start_node.coords.copy()
        else:
            if self.verbose>1:
                print('Adding first segment to branch {}'.format(branch.id))
            parent = None
            pdirection = None
            
        if parent is not None:
            if len(parent.end_node.downstream_segments)>0 and parent.end_node.downstream_segments[0].branch.id==branch.id:
                branch.update() # Hack...!
        #elif parent is None and branch.id!=0:
        #    import pdb
        #    pdb.set_trace()
            
        # If end coordinate is not provided, try and use direction and length instead
        if end_coord is None and direction is not None and length is not None and start_coord is not None:
            end_coord = start_coord + direction*length
                
        #print('Add segment, diameter: {}, seg length: {}, dir: {}, pdir: {}'.format(diameter,length,direction,pdirection))
            
        segment,code = self.create_segment(start_node=start_node,parent=parent,diameter=diameter,branch=branch,**kwargs)
        if segment is None:
            if self.verbose>1:
                print('Segment could not be created: {}'.format(code))
            return None,code
        
        if segment.start_node is None and parent is None:
            #if inside_domain_function is not None:
            #    if not inside_domain_function(start_coord,segment=branch.tip_segment,diameter=diameter):
            #        code = 'outside_domain'
            #        if self.verbose>1:
            #            print('Segment start node could not be created: {}'.format(code))
            #        return None,code
            start_node,code = self.add_node(coords=start_coord,type='root')
            if start_node is None:
                if self.verbose>1:
                    print('Segment start node could not be created: {}'.format(code))
                return None,code
            segment.set_property('start_node',start_node)
        elif segment.start_node is None and parent is not None:
            segment.set_property('start_node',start_node)
            
        if segment.end_node is None:
            #print('Add segment, new node: {}'.format(end_coord))
            #if inside_domain_function is not None:
            #    if not inside_domain_function(end_coord,segment=branch.tip_segment,start_coord=start_coord,branch=branch,diameter=diameter):
            #        code = 'outside_domain'
            #        if self.verbose>1:
            #            print('Segment end node could not be created: {}'.format(code))
            #        return None,code
            end_node,code = self.add_node(coords=end_coord,ignore_domain=ignore_domain)
            if end_node is None:
                if self.verbose>1:
                    print('Segment end node could not be created: {}'.format(code))
                return None,code
            segment.set_property('end_node',end_node)
            
        segment.set_property('fixed_diameter',diameter)

        if 'normal' in locals() and normal is not None :
            segment.set_property('normal',normal)
        elif 'normal' in kwargs.keys() and kwargs['normal'] is not None :
            segment.set_property('normal',kwargs['normal'])
        elif parent is not None:
            normal = segment.propogate_normal(pdirection,segment.direction,pnormal)
            #R = geometry.align_vectors(pdirection,segment.direction)
            #normal = np.dot(R,pnormal)
            #if np.abs(np.rad2deg(np.arccos(np.dot(normal,pnormal))))>10.:
            #    import pdb
            #    pdb.set_trace()
            #    normal = segment.propogate_normal(pdirection,segment.direction,pnormal)
            segment.set_property('normal',normal)
            
        if segment.status!='active':
            if self.verbose>1:
                print('Segment initialisation error...')
            return None, 'error'
             
        #if len(branch.segments)==0 and branch.parent is not None:
        #    branch.parent.tip_segment.associate_daughter(segment)
        #    segment.parent = branch.parent.tip_segment
            
        segment.associate_branch(branch)
        if branch.tip_segment is not None and branch.tip_segment!=segment:
            branch.tip_segment.end_node.associate_downstream_segment(segment)
            if segment.parent is None:
                segment.set_property('parent',branch.tip_segment)
        if segment.start_node=='end':
            segment.start_node = 'inline'
        if parent is not None:
            parent.end_node.associate_downstream_segment(segment)
        branch.associate_segment(segment)
        
        if parent is not None and len(self.inherited_segment_property)>0:
            for prop in self.inherited_segment_property:
                try:
                    propname = prop['name']
                    propval = getattr(parent,propname)
                    #inherits.append({'name':propname,'value':propval})
                    setattr(segment,propname,propval)
                except Exception as e:
                    print(e)
        
        if self.verbose>1:
            print('Branch {} contains {} segments, and is {:.2f} um long'.format(branch.id,len(branch.segments),branch.get_length()))
            
        self.add_segment(segment)

        return segment, ''   
       
    def plot(self,isotropic=True,plot_nodes=True,show=True,ax=None,plot_normals=False,plot_terminals=True,terminals=None,highlight_branches=[],highlight_segments=[],highlight_nodes=[]):
    
        def update_plot_range(plot_range,coord):
            for i in range(3):
                if plot_range[i][0] is None or coord[i]<plot_range[i][0]:
                    plot_range[i][0] = coord[i]
                if plot_range[i][1] is None or coord[i]>plot_range[i][1]:
                    plot_range[i][1] = coord[i]
            return plot_range

        plot_range = [[None,None],[None,None],[None,None]]

        for seg in self.segments:
        
            # Plot edge points
            points = seg.get_edge_coords()
            if seg in highlight_segments or seg.id in highlight_segments:
                edge_col = 'blue'
            elif seg.branch in highlight_branches or seg.branch.id in highlight_branches:
                edge_col = 'red'
            else:
                edge_col = 'black'
            for i,pts in enumerate(points):
                plot_range = update_plot_range(plot_range,pts)
                if i>0:
                    ax = plot_3d_vector(points[i-1],points[i],ax=ax,c=edge_col)
                    
            # Plot normals
            if plot_normals:
                normals = seg.get_edge_normals()
                norm_len = norm(points[1]-points[0])
                for i,pts in enumerate(normals):
                    v0,v1 = points[i],points[i]+normals[i]*norm_len
                    ax = plot_3d_vector(v0,v1,ax=ax,c='grey')
                
            # Plot nodes    
            if plot_nodes:
                pts = seg.start_node.coords
                 
                def plot_colour(node_type):
                    if node_type=='end':
                        col = 'blue'
                    elif node_type=='root':
                        col = 'orange'
                    elif node_type=='inline':
                        col = 'red'
                    elif node_type=='branch':
                        col = 'green'
                    elif node_type=='orphan':
                        col = 'yellow'
                    else:
                        col = 'black'
                    return col
                          
                node_type = seg.start_node.type
                col = plot_colour(node_type)
                ax.scatter(pts[0],pts[1],pts[2],c=col)
                pts = seg.end_node.coords
                node_type = seg.end_node.type
                col = plot_colour(node_type)               
                ax.scatter(pts[0],pts[1],pts[2],c=col)
            
        plot_range = np.asarray(plot_range)
        
        #Plot terminals
        if plot_terminals:
            # Terminals provided as keyword
            if terminals is not None:
                for terminal in terminals:
                    ax.scatter(terminal[0],terminal[1],terminal[2],c='y')
                    
            # Terminals attached to branches
            for branch in self.branches:
                if branch.terminal is not None:
                    ax.scatter(branch.terminal[0],branch.terminal[1],branch.terminal[2],c='b')
            
        if isotropic:
            set_axes_equal(ax)
            
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if show:
            plt.show()
        return ax  
        
#    def add_branch_to_node(self,node,parent_branch,**kwargs): 
#        branch = self.associate_branch(parent=parent_branch,**kwargs)
#        return branch

    def get_root_nodes(self):
        return [x for x in self.nodes if x.type=='root'] 
        
                
    def remove_intermediate_nodes(self):
    
        node_list = self.nodes.copy()
        rem_node, rem_seg = [], []
        conns,segments,paths = [],[],[]
        
        for node in node_list:
            #if len(node.downstream_segments)!=1 or len(node.downstream_segments)==0 or len(node.upstream_segments)!=1: # branch, root or terminal             
            if node.type in ['root','end','branch']:

                for seg in node.downstream_segments:
                    
                    # Find downstream branch/terminal node
                    end_node = None
                    visited_node, visited_seg, start_seg = [seg.start_node],[],seg
                    cur_seg = seg
                    error = False
                    while True:
                    
                        visited_seg.append(cur_seg)
                        visited_node.append(cur_seg.end_node)
                        
                        if cur_seg.branch.status=='removed': # Shouldn't be necessary...
                            error = True
                            print('Removed branch detected! {}'.format(cur_seg.branch.id))
                            break
                        # If this isn't the first segment, it then the current segment needs to be flagged for removal
                        if cur_seg!=seg and cur_seg not in rem_seg:
                            rem_seg.append(cur_seg)
                    
                        # If we've found a branch or terminal at the end of the current segment, then break the loop
                        #if len(cur_seg.end_node.downstream_segments)>1 or len(cur_seg.end_node.downstream_segments)==0:
                        if cur_seg.end_node.type in ['root','end','branch']:
                            end_node = cur_seg.end_node
                            if end_node is None:
                                error = True
                            break
                        else:
                            # Flag the current segment end node for removal
                            if cur_seg.end_node not in rem_node:
                                rem_node.append(cur_seg.end_node)
                                
                            # Assign new segment for next iteration (should be exactly one downstream segment available)
                            cur_seg = cur_seg.end_node.downstream_segments[0]
                            if cur_seg.end_node in visited_node: # Check we've not been here before and are in a loop
                                breakpoint()
                                error = True
                                break
                            if cur_seg.branch.status=='removed': # Check we are in a viable branch
                                error = True
                                break
                        
                    if not error:
                        conns.append([node,end_node])
                        segments.append(start_seg)
                        paths.append(visited_seg)
            else:
                rem_node.append(node)

        for i,conn in enumerate(conns):

            seg, seg_path = segments[i],paths[i]
            assert seg.start_node==conn[0]
            assert seg==seg_path[0]

            # Reassign the first segment in the path to link branch points         
            for j,cur_seg in enumerate(seg_path):
                edge = cur_seg.get_edge_coords()
                diameters = cur_seg.diameters
                normals = cur_seg.normals
                if j==0:
                    path = [x for x in edge]
                    path_normals = [x for x in normals]
                    path_diameters = [x for x in diameters]
                else:
                    path.extend([x for k,x in enumerate(edge) if k>0])
                    path_normals.extend([x for k,x in enumerate(normals)])
                    path_diameters.extend([x for k,x in enumerate(diameters) if k>0])
                    
            path = arr(path)
            path_normals = arr(path_normals)
            path_diameters = arr(path_diameters)

            assert np.all(conn[0].coords==path[0])
            assert np.all(conn[1].coords==path[-1])
            
            seg.end_node = conn[1]
            seg.set_edge_points(path)
            seg.set_property('diameters',path_diameters)
            seg.set_property('normals',path_normals)
            seg.update()
            
            # Delete segments
            rnodes = []
            for j,seg in enumerate(seg_path):
                if j>0:
                    if seg not in rem_seg:
                        breakpoint()
                    try:
                        self.remove_segment(seg,propogate=False)
                    except Exception as e:
                        print(e)
                        #breakpoint()
                    rnodes.append(seg.start_node)
            for node in rnodes:
                self.remove_node(node)
        
    # UTILITIES
    
    def get_branch_axes(self,branch_direction,branch_normal):
        #return xaxis,yaxis,zaxis
        # Branch z-axis (running along branch) in world frame
        zaxis_b = branch_direction 
        zaxis_b /= np.linalg.norm(zaxis_b)
        
        # Branch x-axis in world frame
        xaxis_b = branch_normal
        xaxis_b /= np.linalg.norm(xaxis_b)

        # Branch y-axis in world frame
        yaxis_b = np.cross(xaxis_b,zaxis_b) 
        try:
            yaxis_b /= np.linalg.norm(yaxis_b)
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()
        yaxis_b = -yaxis_b
        
        return xaxis_b,yaxis_b,zaxis_b  
        
    def world_to_branch(self,v,branch_direction,branch_normal):
        xaxis_b,yaxis_b,zaxis_b = self.get_branch_axes(branch_direction,branch_normal)
        R = np.eye(3)
        R[0:3,0] = xaxis_b
        R[0:3,1] = yaxis_b
        R[0:3,2] = zaxis_b
        return np.dot(np.linalg.inv(R),v)        
        
    def branch_to_world(self,v,branch_direction,branch_normal):       
        xaxis_b,yaxis_b,zaxis_b = self.get_branch_axes(branch_direction,branch_normal)
        R = np.eye(3)
        R[0:3,0] = xaxis_b
        R[0:3,1] = yaxis_b
        R[0:3,2] = zaxis_b
        return R.dot(v)        
        
    # I/O
    
    def import_amira(self, gfile):
    
        from pymira import spatialgraph
        sg = spatialgraph.SpatialGraph(initialise=True)
        sg.read(gfile)
        
        node_coords = sg.get_data('VertexCoordinates')
        conns = sg.get_data('EdgeConnectivity')
        nedgepoints = sg.get_data('NumEdgePoints')
        edges = sg.get_data('EdgePointCoordinates')
        nep_total = np.sum(nedgepoints)
        
        nodes,node_conn_count = [],[]
        for i,node in enumerate(node_coords):
            new_node = Node(coords=node)
            self.add_node(new_node)
            nodes.append(new_node)
            
            # Find end and branching nodes
            sind = np.where((conns[:,0]==i) | (conns[:,1]==i))
            node_conn_count.append(len(sind[0]))
        
        rad_names = ['Radius','Radii','radius','radii','thickness']
        for rad_name in rad_names:
            radii = sg.get_data(rad_name)
            if radii is not None:
                break
        if radii is None:
            breakpoint()
            
        order = sg.get_data('Order')
        if order is None:
            order = np.zeros(nep_total)
            
        category = sg.get_data('Category')
        if category is None:
            category = np.zeros(nep_total)
            
        quadrant = sg.get_data('quadrant')
        if quadrant is None:
            quadrant = np.zeros(nep_total)
            
        branch = sg.get_data('branch')
        if branch is None:
            branch = np.zeros(nep_total)
        
        segs,branch_ids,branches = [],[],[]
        for i,conn in enumerate(conns):
        
            n0,n1 = nodes[conn[0]], nodes[conn[1]]
            nep = nedgepoints[i]
            x0 = np.sum(nedgepoints[0:i])
            x1 = x0 + nep
            edge = edges[x0:x1]
            diameters = radii[x0:x1] * 2
            
            branch_id = int(branch[x0:x1][0])
            if branch_id not in branch_ids:
                branch_ids.append(branch_id)
                cur_branch = Branch()
                self.add_branch(cur_branch)
                branches.append(cur_branch)
            else:
                cur_branch = branches[branch_id]
            
            new_seg = Segment(start_node=n0,end_node=n1,diameters=diameters,branch=cur_branch,coords=edge)
            self.add_segment(new_seg,keep_tip=True)
            cur_branch.associate_segment(new_seg)

        # Sort out branches
        for i,branch in enumerate(self.branches):
            cur_segs = branch.segments
            
            # Root segment
            roots = [x for x in cur_segs if len(x.start_node.upstream_segments)==0 or len([y for y in x.start_node.upstream_segments if y.branch!=branch])>0]
            if len(roots)==1:
                branch.root_segment = roots[0]
                root_node = roots[0].start_node
                nc = node_conn_count[root_node.id]
            else:
                breakpoint()
                
            # Tip segment
            tips = [x for x in cur_segs if len(x.end_node.downstream_segments)==0 or np.all([y.branch!=branch for y in x.end_node.downstream_segments])]
            if len(tips)==1:
                branch.root_segment = tips[0]
            else:
                breakpoint()
                
            branch.update()
                
            # TODO: Daughters
    
    def write_amira(self,gfile):
        
        nnodes = len(self.nodes)
        #node_ids = self.node_id[ni]
        #new_node_ids = np.linspace(0,nnode-1,nnode,dtype='int')
        #node_ids = np.linspace(0,nnode-1,nnode,dtype='int')

        nseg = len(self.segments)
        segments, points, npoint, categories, types, diameters, distances = self.get_edge_points()
        npoints_total = np.sum(npoint)
        
        nodes = self.get_node_coords()
        conns = np.zeros([nseg,2],dtype='int')
        for i,seg in enumerate(self.segments):
            conns[i,0] = self.nodes.index(seg.start_node)
            conns[i,1] = self.nodes.index(seg.end_node)
            
        order, branch_id, cycle, density, node_fitness, branch_fitness, category = [],[],[],[],[],[],[]
        scalars = [[] for i in range(len(self.graph_params))]
        
        for seg in self.segments:

            nc = seg.coords.shape[0]
            cat_id = 0
            if seg.branch.category=='artery':
                cat_id = 1
            elif seg.branch.category=='vein':
                cat_id = 2

            for i,e in enumerate(self.graph_params):
                if e['name'].lower() in ['radius','radii']:
                    vals = seg.diameters/2.
                    scalars[i].extend(vals)
                elif e['name'].lower() in ['vessel_type','vessel type','category']:
                    str_vals = getattr(seg.branch,'category')
                    v = str_vals
                    if v=='artery':
                        vals = np.zeros(nc,dtype='int') + 0
                    elif v=='vein':
                        vals = np.zeros(nc,dtype='int') + 1
                    else:
                        vals = np.zeros(nc,dtype='int') + 2
                                
                    scalars[i].extend(vals)
                    
                elif e['name'].lower() in ['branch']:
                    vals = np.zeros(nc,dtype='int') + getattr(seg.branch,'id')       
                    scalars[i].extend(vals)
                elif e['name']=='branch_parent':
                    #breakpoint()
                    parent = seg.branch.parent
                    if parent is not None:
                        vals = np.zeros(nc,dtype='int') + parent.id
                    else:
                        vals = np.zeros(nc,dtype='int') - 1
                    scalars[i].extend(vals)
                    
                elif e['element_type']=='branch':
                    vals = [getattr(seg.branch,e['data_field']) for j in range(nc)]
                    scalars[i].extend(vals)
                
                elif e['element_type']=='node':
                    vals = [getattr(seg.end_node,e['data_field']) for j in range(nc)]
                    scalars[i].extend(vals)
            
        order = arr(order)
        density = arr(density)
        branch_id = arr(branch_id)
        cycle = arr(cycle)
        node_fitness = arr(node_fitness)
        branch_fitness = arr(branch_fitness)
        category = arr(category)
        
        for i,sc in enumerate(scalars):                
            scalars[i] = arr(scalars[i])
            
        """
        Convert a vessel graph to Amira spatial graph format 
        """
        from pymira import spatialgraph
        sg = spatialgraph.SpatialGraph(initialise=True)
        # Fill default fields
        #nnodes = nodes.shape[0]
        nconns = conns.shape[0]
        sg.definitions[0]['size'] = [nnodes]
        sg.definitions[1]['size'] = [nconns]
        sg.definitions[2]['size'] = [npoints_total]
        sg.set_data(nodes,name='VertexCoordinates')
        sg.set_data(conns,name='EdgeConnectivity')
        sg.set_data(npoint,name='NumEdgePoints')
        sg.set_data(points,name='EdgePointCoordinates')
        
        ind = 5 
        for i,e in enumerate(self.graph_params):
            if len(scalars[i])>0:
                field = sg.add_field(name=e['name'],marker='@{}'.format(ind),
                                      definition='POINT',type='float',
                                      nelements=1,nentries=[0],data=None)
                sg.set_data(scalars[i],name=e['name'])
                ind += 1

        sg.add_parameter('InletNodeId',self.nodes.index(self.root_node))
        
        sg.write(gfile) 
        return sg     
