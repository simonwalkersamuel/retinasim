from pymira import spatialgraph, reanimate, combine_graphs, amirajson
import numpy as np
import os
import shutil
join = os.path.join
arr = np.asarray
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
import subprocess

from retinasim import geometry
from retinasim.eye import Eye
from pymira.spatialgraph import update_array_index,delete_vertices

# REANIMATE (flow estimation) directories
REANIMATE_DIR = "/PATH/TO/REANIMATE/DISTRIBUTION" 
REANIMATE_DATA_FILENAME = "/PATH/TO/REANIMATE/DAT_FILE.dat"
REANIMATE_TMP_DIR_LOC = "/PATH/TO/REANIMATE/TEMP/DIRECTORY"

def test_midline_collision(eye,coords,edges,extent):
    x0 = eye.optic_disc_centre
    x1 = eye.fovea_centre
    v = x1-x0
    v = v / np.linalg.norm(v)
    # Distance to edges
    l_x0_A = np.linalg.norm(x0-extent[:,0])
    l_x0_B = np.linalg.norm(x0-extent[:,1])
    A = x0 - v*l_x0_A*1.2
    B = x0 + v*l_x0_B*1.2
    A2,B2 = coords[edges[:,0]], coords[edges[:,1]]
    A[2],B[2],A2[:,2],B2[:,2] = 0.,0.,0.,0.
    test = geometry.segment_intersection_multi(A,B,A2.transpose(),B2.transpose()) 
    return test
    
def set_and_write_graph(graph,data,ofile):
    fields = ['VertexCoordinates','EdgeConnectivity','EdgePointCoordinates','NumEdgePoints','Radii','VesselType','midLinePos']

    if graph is None:
        fields = ['VertexCoordinates','EdgeConnectivity','EdgePointCoordinates','NumEdgePoints','Radii','VesselType','midLinePos']
        graph = spatialgraph.SpatialGraph(initialise=True,scalars=['Radii','VesselType','midLinePos'])
    else:
        fields = ['VertexCoordinates','EdgeConnectivity','EdgePointCoordinates','NumEdgePoints',graph.get_radius_field()['name'],'VesselType','midLinePos']

    for i,field in enumerate(fields):
        graph.set_data(data[i],name=fields[i])

    graph.set_definition_size('VERTEX',data[0].shape[0])
    graph.set_definition_size('EDGE',data[1].shape[0])
    graph.set_definition_size('POINT',data[4].shape[0])
    
    if ofile is not None:
        print(f'Writing graph to {ofile}')
        graph.write(ofile)
            
    graph.set_graph_sizes()
        
    return graph    
    
def get_graph_fields(graph):

    res = []
    res.append(graph.get_data('VertexCoordinates'))
    res.append(graph.get_data('EdgeConnectivity'))
    res.append(graph.get_data('EdgePointCoordinates'))
    res.append(graph.get_data('NumEdgePoints'))
    radField = graph.get_radius_field()
    res.append(radField['data'])
    category = graph.get_data('VesselType')
    if category is None:
        category = graph.get_data('Category')
    if category is None:
        print('Warning, category field could not be located')
        category = radField['data'].copy()
    res.append(category)
    mlp = graph.get_data('midLinePos')
    if mlp is None:
        #print('Warning, mlp field could not be located')
        mlp = radField['data'].copy() * 0.
    res.append(mlp)
    
    return res    
    
def filter_graph_by_radius(graph,min_filter_radius=5.,filter_clip_radius=None,write=False,ofile='',keep_stubs=False,stub_len=100.):

    """
    min_filter_radius: All edges with radii < this value are deleted
    filter_clip_radius: All edges with radii < this value and > min_filter_radius are set to filter_clip_radius    
    """

    nodecoords, edgeconn, edgepoints, nedgepoints, radius, category, mlp = get_graph_fields(graph)

    # List all edge indices
    inds = np.linspace(0,edgeconn.shape[0]-1,edgeconn.shape[0],dtype='int')
    # Define which edge each edgepoint belongs to
    edge_inds = np.repeat(inds,nedgepoints)
    if filter_clip_radius is not None:
        clip_edge_inds = np.where((radius>=filter_clip_radius) & (radius<=min_filter_radius))
        radius[clip_edge_inds] = min_filter_radius
    del_edge_inds = np.where(radius<min_filter_radius)
    
    edge_stubs = arr([])
    if keep_stubs:
        node_radius = np.zeros(nodecoords.shape[0]) - 1
        rname = graph.get_radius_field_name()
        rind = [i for i,s in enumerate(graph.get_scalars()) if s['name']==rname][0]
        for i in trange(graph.nedge):
            edge = graph.get_edge(i)
            rads = np.min(edge.scalars[rind])
            node_radius[edge.start_node_index] = np.max([rads,node_radius[edge.start_node_index]])
            node_radius[edge.end_node_index] = np.max([rads,node_radius[edge.end_node_index]])
        
        is_stub = np.zeros(radius.shape[0],dtype='bool')
        stub_loc = np.zeros(graph.nedge,dtype='int') - 1
        for i in trange(graph.nedge):
            edge = graph.get_edge(i)   
            rads = edge.scalars[rind]
            #epointinds = np.linspace(edge.i0,edge.i1,edge.npoints,dtype='int')
            if np.any(rads<min_filter_radius):
                if node_radius[edge.start_node_index]>min_filter_radius:
                    is_stub[edge.i0] = True
                    stub_loc[i] = 0
                elif node_radius[edge.end_node_index]>min_filter_radius:
                    is_stub[edge.i0] = True   
                    stub_loc[i] = 1

        del_edge_inds = np.where((radius<min_filter_radius) & (is_stub==False))
        edgepoint_edges = np.repeat(np.linspace(0,edgeconn.shape[0]-1,edgeconn.shape[0],dtype='int'),nedgepoints)
        edge_stubs = np.unique(edgepoint_edges[is_stub])
        #breakpoint()
        
        edges = edgeconn[edge_stubs]
        stub_loc = stub_loc[edge_stubs]
        # Shorten stubs
        edgepoints_valid = np.ones(edgepoints.shape[0],dtype='bool') 
        for i,edge in enumerate(tqdm(edges)):
             nodes = nodecoords[edge]
             edgeObj = graph.get_edge(edge_stubs[i])
             points = edgeObj.coordinates
             if stub_loc[i]==0:
                 lengths = np.concatenate([[0.],np.cumsum(np.linalg.norm(points[1:]-points[:-1]))])
                 if np.max(lengths)>=stub_len:
                     x0 = points[0]
                     x1 = points[lengths>=stub_len]
                     clen = stub_len
                 else:
                     x0 = points[0]
                     x1 = points[-1]
                     clen = np.linalg.norm(x1-x0)
                 vn = x1 - x0
                 vn = vn / np.linalg.norm(x1-x0)
                 nodecoords[edge[1]] = x0 + vn*stub_len
                 new_edgepoints = [x0,x1]
             elif stub_loc[i]==1:
                 points = np.flip(points,axis=0)
                 lengths = np.concatenate([[0.],np.cumsum(np.linalg.norm(points[1:]-points[:-1]))])
                 if np.max(lengths)>=stub_len:
                     x0 = points[0]
                     x1 = points[lengths>=stub_len]
                     clen = stub_len
                 else:
                     x0 = points[0]
                     x1 = points[-1]
                     clen = np.linalg.norm(x1-x0)
                 vn = x1 - x0
                 vn = vn / np.linalg.norm(x1-x0)
                 nodecoords[edge[0]] = x0 + vn*clen
                 new_edgepoints = [x1,x0]
             else:
                 breakpoint()
             edgepoints[edgeObj.i0] = new_edgepoints[0]
             edgepoints[edgeObj.i0+1] = new_edgepoints[1]
             if edgeObj.npoints>2:
                 edgepoints_valid[edgeObj.i0+1:edgeObj.i0] = False
             nedgepoints[i] = 2
        #breakpoint()
        edgepoints = edgepoints[edgepoints_valid]

    print(f'{len(clip_edge_inds[0])} edges with radii>{filter_clip_radius} and radii<{min_filter_radius} clipped.')
    print(f'{len(del_edge_inds[0])} edges with radii<{min_filter_radius} clipped.')

    # Find unique edge index references
    keep_edge = np.ones(edgeconn.shape[0],dtype='bool')
    # Convert to segments
    del_inds = np.unique(edge_inds[del_edge_inds])
    keep_edge[del_inds] = False
    keep_inds = np.where(keep_edge)[0]
    # Define nodes to keep positively (i.e. using the keep_inds rather than del_inds) so that nodes are retained that appear in edges that aren't flagged for deletion
    node_keep_inds = np.unique(edgeconn[keep_inds].flatten())
    keep_node = np.zeros(nodecoords.shape[0],dtype='bool')
    keep_node[node_keep_inds] = True
    
    graph = set_and_write_graph(graph,[nodecoords,edgeconn,edgepoints,nedgepoints,radius,category,mlp],None)
    graph = delete_vertices(graph,keep_node)
    
    if write:
        graph.write(ofile)  
        
    return graph  
        
def interpolate_graph(graph,**kwargs):
    ed = spatialgraph.Editor()
    ed.interpolate_edges(graph,**kwargs)
    return graph        
    
def get_node_count(graph,edge_node_lookup=None,restore=False,tmpfile=None,graph_params=None):

    if graph is not None:
        nodecoords_new,edgeconn_new,edgepoints_new,nedgepoints_new,radius_new,category_new,mlp_new = get_graph_fields(graph)
    elif graph_params is not None:
        nodecoords_new,edgeconn_new,edgepoints_new,nedgepoints_new,radius_new,category_new,mlp_new = graph_params
    else:
        return None
    
    # Which edge each node appears in
    if edge_node_lookup is not None:
        #edge_node_lookup = create_edge_node_lookup(nodecoords_new,edgeconn_new,tmpfile=tmpfile,restore=restore)
        node_count = arr([len(edge_node_lookup[i]) for i in range(nodecoords_new.shape[0])])
    else:
        unq,count = np.unique(edgeconn_new,return_counts=True)
        all_nodes = np.linspace(0,nodecoords_new.shape[0]-1,nodecoords_new.shape[0],dtype='int')
        node_count = np.zeros(nodecoords_new.shape[0],dtype='int') 
        node_count[np.in1d(all_nodes,unq)] = count
    return node_count    
    
    
def remove_endpoint_nodes(graph,keep=None,return_nodes=False):

    nodecoords_new,edgeconn_new,edgepoints_new,nedgepoints_new,radius_new,category_new,mlp_new = get_graph_fields(graph)
    
    # Which edge each node appears in
    #tmpfile = '/mnt/data2/retinasim/cco/graph/tmp.p'
    node_count = get_node_count(graph) #tmpfile=tmpfile,restore=False)
    
    keep_node = np.ones(node_count.shape[0],dtype='bool')
    keep_node[node_count<=1] = False
    if keep is not None:
        #keepF = arr([x for x in keep if x!=None])
        try:
            keep_node[arr([x for x in keep if x!=None])] = True
        except Exception as e:
            print(e)
            breakpoint()
            
    nodes = graph.get_data('VertexCoordinates').copy()

    graph,lookup = delete_vertices(graph,keep_node,return_lookup=True)
    
    if return_nodes==False:
        return lookup
    else:
        return lookup, nodes[keep_node==False]

def remove_all_endpoints(graph,keep=None,max_it=1000,return_nodes=False):

    if keep is None:
        nkeep = 0
    else:
        keep = arr([x for x in keep if x!=None])
        nkeep = len(keep)

    ep_nodes = []
    for i in range(max_it):
        # Number of endpoints before removal
        node_count = get_node_count(graph)
        endpoints = np.where(node_count==1)
        nendpoint_pre = len(endpoints[0])
        
        # Remove endpoints
        lookup,del_nodes = remove_endpoint_nodes(graph,keep=keep,return_nodes=True)
        # Update keep node indices
        keep = lookup[keep]
        ep_nodes.extend(del_nodes)
        
        # Number of endpoints after removal
        node_count = get_node_count(graph)
        endpoints = np.where(node_count==1)
        nendpoint_post = len(endpoints[0])
        #print(f'Remove endpoints: was {nendpoint_pre}, is now {nendpoint_post}. Nnodes: {graph.nnode}')
        if nendpoint_post==nkeep:
            break
        if nendpoint_post<nkeep:
            print('Error, too many endpoints deleted!')
            break
    
    if return_nodes==False:       
        return graph
    else:
        return graph, ep_nodes
    
def identify_endpoints(graph):
    node_count = get_node_count(graph)
    endpoints = np.where(node_count==1)[0]
    return endpoints    
    
def create_edge_node_lookup(nodecoords_new,edgeconn_new,tmpfile=None,restore=False):

    verts_used = np.unique(edgeconn_new)
    srt0 = np.argsort(edgeconn_new[:,0])
    bc0 = np.bincount(edgeconn_new[:,0][srt0])
    srt1 = np.argsort(edgeconn_new[:,1])
    bc1 = np.bincount(edgeconn_new[:,1][srt1])
    
    npoints = nodecoords_new.shape[0]
    
    # Make equal length
    if bc0.shape[0]<npoints: #<bc1.shape[0]:
       dif = npoints-bc0.shape[0] #bc1.shape[0]-bc0.shape[0]
       bc0 = np.append(bc0,np.zeros(dif,dtype='int'))
    if bc1.shape[0]<npoints: #bc0.shape[0]:
       dif = npoints-bc1.shape[0] # bc0.shape[0]-bc1.shape[0]
       bc1 = np.append(bc1,np.zeros(dif,dtype='int'))

    # Find edges incorporating each node
    point_edge_inds = [[] for i in range(npoints)]
    for pnt_ind in trange(npoints):
        i0 = np.sum(bc0[:pnt_ind])
        i1 = i0 + bc0[pnt_ind]
        simp_inds_0 = srt0[i0:i1]
        test0 = np.all(edgeconn_new[:,0][srt0[i0:i1]]==pnt_ind)
        
        i0 = np.sum(bc1[:pnt_ind])
        i1 = i0 + bc1[pnt_ind]
        simp_inds_1 = srt1[i0:i1]
        test1 = np.all(edgeconn_new[:,1][srt1[i0:i1]]==pnt_ind)
        
        point_edge_inds[pnt_ind] = np.concatenate([simp_inds_0,simp_inds_1])
        
        if not test1 or not test0:
            breakpoint

    return point_edge_inds
    
def identify_inlet_outlet(graph,tmpfile=None,restore=False,ignore=None):

    nodecoords, edgeconn, edgepoints, nedgepoints, radius, category, mlp = get_graph_fields(graph)

    inds = np.linspace(0,edgeconn.shape[0]-1,edgeconn.shape[0],dtype='int')
    edge_inds = np.repeat(inds,nedgepoints)
    first_edgepoint_inds = np.concatenate([[0],np.cumsum(nedgepoints)[:-1]])

    #edge_node_lookup = create_edge_node_lookup(nodecoords,edgeconn,tmpfile=tmpfile,restore=restore)
            
    # Calculate node connectivity
    node_count = get_node_count(graph) #,edge_node_lookup=edge_node_lookup)
    # Find graph end nodes (1 connection only)
    term_inds = np.where(node_count==1)
    terminal_node = np.zeros(nodecoords.shape[0],dtype='bool')
    terminal_node[term_inds] = True

    # Assign a radius to nodes using the largest radius of each connected edge
    edge_radius = radius[first_edgepoint_inds]

    # Assign a category to each node using the minimum category of each connected edge (thereby favouring arteries/veins (=0,1) over capillaries (=2))
    edge_category = category[first_edgepoint_inds]
    
    # Locate arterial input(s)
    mask = np.ones(edgeconn.shape[0])
    mask[(edge_category!=0) | ((node_count[edgeconn[:,0]]!=1) & (node_count[edgeconn[:,0]]!=1))] = np.nan
    if ignore is not None:
        mask[(np.in1d(edgeconn[:,0],ignore)) | (np.in1d(edgeconn[:,1],ignore))] = np.nan
    if np.nansum(mask)==0.:
        a_inlet_node = None
    else:
        a_inlet_edge_ind = np.nanargmax(edge_radius*mask)
        a_inlet_edge = edgeconn[a_inlet_edge_ind]
        a_inlet_node = a_inlet_edge[node_count[a_inlet_edge]==1][0]
    
    # Locate vein output(s)
    mask = np.ones(edgeconn.shape[0])
    mask[(edge_category!=1) | ((node_count[edgeconn[:,0]]!=1) & (node_count[edgeconn[:,0]]!=1))] = np.nan
    if ignore is not None:
        mask[(np.in1d(edgeconn[:,0],ignore)) | (np.in1d(edgeconn[:,1],ignore))] = np.nan
    if np.nansum(mask)==0.:
        v_outlet_node = None
    else:
        v_outlet_edge_ind = np.nanargmax(edge_radius*mask)
        v_outlet_edge = edgeconn[v_outlet_edge_ind]
        v_outlet_node = v_outlet_edge[node_count[v_outlet_edge]==1][0]
    
    return a_inlet_node,v_outlet_node   
    
def join_feeding_vessels(graph, offset=arr([-10.,-10.,-1000])):
    """
    Retina networks are usually created with separate netowrks for above and below the midline
    This function joins the arterial inlets and venous outlets together
    """

    # Find the inlets and outlets
    a_inlet_node,v_outlet_node = identify_inlet_outlet(graph)
    a_inlet_node1,v_outlet_node1 = identify_inlet_outlet(graph,ignore=[a_inlet_node,v_outlet_node])
    if a_inlet_node1 is None:
       print('Warning: one or fewer arterial inlets located!')
    if v_outlet_node1 is None:
       print('Warning: one or fewer venous outlets located!')
    a_inlets = arr([a_inlet_node,a_inlet_node1])
    v_outlets = arr([v_outlet_node,v_outlet_node1])
    
    def connect_ends(endpoint_nodes,graph,offset=arr([-10.,-10.,-1000])):
    
        # Get standard fields and create copies for editing
        nodecoords, edgeconn, edgepoints, nedgepoints, radius, category, mlp = get_graph_fields(graph)
        radName = graph.get_radius_field()['name']
        
        a_coords = nodecoords[endpoint_nodes]
    
        # Create arterial inlet node and edges connecting to original inlets
        a0 = np.mean(a_coords,axis=0)
        nodecoords_new = np.vstack([nodecoords,a0])
        a_node_ind = nodecoords_new.shape[0]-1
        a0start = a0 + offset
        #print(a0)
        #print(a0start)
        nodecoords_new = np.vstack([nodecoords_new,a0start])
        a_start_node_ind = nodecoords_new.shape[0]-1
        
        # Connect new start point to mean point between endnodes
        edgeconn_new = np.vstack([edgeconn,[a_start_node_ind,a_node_ind]])
        # Connect meanpoint to first endpoint 
        edgeconn_new = np.vstack([edgeconn_new,[a_node_ind,endpoint_nodes[0]]])
        # Connect meanpoint to second endpoint 
        edgeconn_new = np.vstack([edgeconn_new,[a_node_ind,endpoint_nodes[1]]])
        
        #print([a_start_node_ind,a_node_ind])
        #print([a_node_ind,endpoint_nodes[0]])
        #print([a_node_ind,endpoint_nodes[1]])
        
        nedgepoints_new = np.concatenate([nedgepoints,[2]])
        nedgepoints_new = np.concatenate([nedgepoints_new,[2]])
        nedgepoints_new = np.concatenate([nedgepoints_new,[2]])
        
        edgepoints_new = np.vstack([edgepoints,[a0start]])
        edgepoints_new = np.vstack([edgepoints_new,[a0]])
        edgepoints_new = np.vstack([edgepoints_new,[a0]])
        edgepoints_new = np.vstack([edgepoints_new,[nodecoords_new[endpoint_nodes[0]]]])
        edgepoints_new = np.vstack([edgepoints_new,[a0]])
        edgepoints_new = np.vstack([edgepoints_new,[nodecoords_new[endpoint_nodes[1]]]])
        
        # Get edges connected to existing endpoints
        a1_edges = graph.get_edges_containing_node(endpoint_nodes[0])
        a2_edges = graph.get_edges_containing_node(endpoint_nodes[1])
        
        # Makes sure they are true endpoints...
        if len(a1_edges)!=1 or len(a2_edges)!=1:
            print('Error, inlets are connected to more than one node!')
            return graph
        
        # Reference the single edge connected to each endpoint        
        edge1 = graph.get_edge(a1_edges[0])
        edge2 = graph.get_edge(a2_edges[0])
        
        # Deal with point radii
        rads1 = edge1.scalars[edge1.scalarNames.index(radName)]
        rads2 = edge2.scalars[edge2.scalarNames.index(radName)]
        new_rad = np.max(np.concatenate([rads1,rads2])) * 1.5
        radius_new = np.concatenate([radius,[new_rad],[new_rad]])
        new_rad = np.max(np.concatenate([rads1,rads2])) * 1.3
        radius_new = np.concatenate([radius_new,[new_rad],[new_rad]])
        radius_new = np.concatenate([radius_new,[new_rad],[new_rad]])
        
        # Deal with point category
        cat1 = edge1.scalars[edge1.scalarNames.index('VesselType')]
        new_cat = cat1[0]
        category_new = np.concatenate([category,[new_cat],[new_cat]])
        category_new = np.concatenate([category_new,[new_cat],[new_cat]])
        category_new = np.concatenate([category_new,[new_cat],[new_cat]])
        
        mlp_new = category_new.copy()
        
        scalars = graph.get_scalars()
        for i,sc1 in enumerate(edge1.scalars):
            sc2 = edge2.scalars[i]
            if edge1.scalarNames[i] not in [radName,'VesselType']:
                if sc1.dtype in [np.float32,np.float64]:
                    new_val = np.mean(sc1) + np.mean(sc2)
                elif sc1.dtype in [np.int32,np.int64]:
                    new_val = sc1[0]
                else:
                    breakpoint()
                    
                curField = [x for x in scalars if x['name']==edge1.scalarNames[i]][0]
                data = curField['data']
                data = np.concatenate([data,[new_val],[new_val]])
                data = np.concatenate([data,[new_val],[new_val]])
                data = np.concatenate([data,[new_val],[new_val]])
                graph.set_data(data,name=curField['name'])
                
        graph = set_and_write_graph(graph, [nodecoords_new,edgeconn_new,edgepoints_new,nedgepoints_new,radius_new,category_new,mlp_new],None)
        return graph
    
    if not any(a_inlets==None):
        graph = connect_ends(a_inlets,graph,offset=offset)
    if not any(v_outlets==None):        
        graph = connect_ends(v_outlets,graph,offset=offset)

    return graph  
    
def split_artery_vein_graphs(graph):

    vt = graph.get_data('VesselType')
    
    # Node vessel type
    vtypeNode = np.zeros(graph.nnode,dtype='int') - 1
    vind = [i for i,s in enumerate(graph.get_scalars()) if s['name']=='VesselType'][0]
    for i in range(graph.nedge):
        edge = graph.get_edge(i)
        vt = int(edge.scalars[vind][0])
        vtypeNode[edge.start_node_index] = vt
        vtypeNode[edge.end_node_index] = vt
    
    s_art = np.where(vtypeNode==0)
    node_del_flag = np.zeros(graph.nnode,dtype='bool')
    node_del_flag[s_art] = True
    vgraph = graph.copy()
    vgraph = delete_vertices(vgraph,~node_del_flag,return_lookup=False)
    
    s_vein = np.where(vtypeNode==1)
    node_del_flag = np.zeros(graph.nnode,dtype='bool')
    node_del_flag[s_vein] = True
    agraph = graph.copy()
    agraph = delete_vertices(agraph,~node_del_flag,return_lookup=False)
    
    return agraph,vgraph
    
def make_dir(dir,overwrite_existing=False):

    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
        elif overwrite_existing:
            shutil.rmtree(dir)
            os.makedirs(dir)
    except Exception as e:
        print(e)

def reanimate_sim(graph,opath=None,ofile=None,a_pressure=80.,v_pressure=30.,quiet=False,reanimate_dir=None,dfile=None):

    """
    Perform REANIMATE flow simulation
    Default paths are defined at top of file
    """

    tmpDir = None
    if reanimate_dir is None:
        reanimate_dir = REANIMATE_DIR
    if dfile is None:
        tmpDir = REANIMATE_TMP_DIR_LOC
        if not os.path.exists(tmpDir):
            os.mkdir(tmpDir)
        while True:   
            tmpDir = join(tmpDir,f'reanimate_temp_{str(int(np.random.uniform(0,10000000))).zfill(8)}')
            if not os.path.exists(tmpDir):
                os.mkdir(tmpDir)
                break
        dfile = join(tmpDir,'reanimate.dat')
        build_path = join(tmpDir,'reanimate') + os.path.sep

    try:
        print(f'Writing dat file... {dfile}')
        reanimate.export_dat(graph,dfile,remove_intermediate=False,nbc=2,pressure=[a_pressure,v_pressure])
        
        print('Calling REANIMATE...')
        p1 = subprocess.call(["./Reanimate",os.path.dirname(dfile)+os.path.sep,os.path.basename(dfile),build_path],cwd=reanimate_dir) #,stdout=subprocess.DEVNULL)
        if p1==0:
            print('REANIMATE complete')
            # Convert result to Amira graph
            fname = join(build_path,'SolvedBloodFlow.txt') #  '/mnt/ml/anaconda_envs/vessel_growth_38/lib/python3.8/site-packages/Reanimate/Build_Data/SolvedBloodFlow.txt'
            graph = reanimate.import_dat(fname)
            
            if opath is not None and ofile is not None:
                print(f'Saving Amira graph to {ofile}')
                graph.write(join(opath,ofile))
        else:
            print('REANIMATE error!')
            breakpoint()
    except Exception as e:
        print(e)

    if tmpDir is not None:
        shutil.rmtree(tmpDir)
        
    return graph       
    

