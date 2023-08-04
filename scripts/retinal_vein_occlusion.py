import numpy as np
from pymira import spatialgraph
from vessel_sim.retina_cco.retina_lsystem import Eye
from tqdm import tqdm, trange
import pickle
import os
join = os.path.join
arr = np.asarray

from retinasim.utility import make_dir
from retinasim.main import flow_ordering
from retinasim.capillary_bed import direct_arterial_venous_connection, voronoi_capillary_bed, remove_arteriole  
from retinasim.diabetic_retinopathy import dbr_iterative, dbr_sim

def match_graphs(graph0,graphN):

    # Given two similar graphs, this finds overlapping ndoes and matched edges

    # Match nodes
    nodes = graph0.get_data('VertexCoordinates').astype('float32')
    nodesN = graphN.get_data('VertexCoordinates').astype('float32')
    nodeLookup = np.zeros(nodesN.shape[0],dtype='int') - 1
    for j,nodeN in enumerate(nodesN):
        mtch = np.where((nodes[:,0]==nodeN[0]) & (nodes[:,1]==nodeN[1]) & (nodes[:,2]==nodeN[2]))
        if len(mtch[0])==1:
            # Lookup maps nodesN indices to node indices
            nodeLookup[j] = mtch[0][0]
        elif len(mtch[0])>1:
            print(f'Warning, match_graphs: degenerate node detected! {j}')
            nodeLookup[j] = mtch[0][0]
        else:
           print(f'Node {j} not present in source graph')

    edgeconn = graph0.get_data('EdgeConnectivity')
    edgeconnN = graphN.get_data('EdgeConnectivity')
    edgeLookup = np.zeros(edgeconnN.shape[0],dtype='int') - 1
    for j,eN in enumerate(edgeconnN):
        # Convert from nodeN to node indices
        elN = nodeLookup[eN]
        if np.all(elN>=0):
            mtch = np.where((edgeconn[:,0]==elN[0]) & (edgeconn[:,1]==elN[1]))
            if len(mtch[0])==1:
                edgeLookup[j] = mtch[0][0]
            elif len(mtch[0])>1:
                print(f'Warning, match_graphs: degenerate edge detected! {j}')
                edgeLookup[j] = mtch[0][0]
        else:
           print(f'Edge {j} not present in source graph')
                
    return nodeLookup, edgeLookup

def retinal_vein_occulsion(gfile=None,cco_path='',output_dir=None,geometry_file=None,recon_only=False,pfile=None,win_width=6000,win_height=6000):

    delta_dir = join(output_dir,'delta_flow')
    if not os.path.exists(delta_dir):
        make_dir(delta_dir)
        
    flow_dir = join(output_dir,'flow')
    if not os.path.exists(flow_dir):
        make_dir(flow_dir)
        
    vessels_dir = join(output_dir,'vessels')
    if not os.path.exists(vessels_dir):
        make_dir(vessels_dir)

    graph = spatialgraph.SpatialGraph()
    graph.read(gfile)
    
    arterial_pressure = 50.
    venous_pressure = 20.
    nstep = 100

    
    eye = Eye()
    eye.load(geometry_file) 
    
    print('Removing degeneracy...')
    deg,_ = graph.test_node_degeneracy()
    if deg:
        ed = spatialgraph.Editor()
        graph = ed.displace_degenerate_nodes(graph,displacement=0.1)

    # Needed to 
    graph0, rem_edge = dbr_sim(eye=eye,graph=graph,nart=0,max_radius=35.,min_displacement=0.,win_width=win_width,win_height=win_height,plot=False,fluorescein=False,write_graph=False,a_pressure=arterial_pressure,v_pressure=venous_pressure)  

    vis = graph.plot_graph(edge_highlight=rem_edge,scalar_color_name='VesselType',log_color=False,show=False,block=False,win_width=win_width,win_height=win_height)
    ofile = join(vessels_dir,f'vesselType.png')
    vis.screen_grab(ofile)
    vis.destroy_window()

    tmppath = '~/temp'
    graphN, rem_edge = dbr_sim(opath=tmppath,eye=eye,graph=graph,nart=1,catId=1,max_radius=70.,min_radius=40.,min_displacement=0.,win_width=win_width,win_height=win_height,plot=False,fluorescein=False,write_graph=False,a_pressure=arterial_pressure,v_pressure=venous_pressure)  
    
    vis = graphN.plot_graph(edge_highlight=rem_edge,scalar_color_name='VesselType',log_color=False,show=False,block=False,win_width=win_width,win_height=win_height)
    ofile = join(vessels_dir,f'vesselType_occ.png')
    vis.screen_grab(ofile)
    vis.destroy_window()

    lookup, edgeLookup = match_graphs(graph0,graphN)

    edgeconn = graph0.get_data('EdgeConnectivity')
    edgeconnN = graphN.get_data('EdgeConnectivity')
    dfl = np.zeros(graphN.nedgepoint)
    filledEdge = np.zeros(graphN.nedge,dtype='bool')
    for j,eN in enumerate(edgeconnN):
        # Convert from nodeN to node indices
        elN = lookup[eN]
        if np.all(elN>=0):

            if edgeLookup[j]>=0:
                edgeN = graphN.get_edge(j)
                edge = graph0.get_edge(edgeLookup[j])
                flInd = edge.scalarNames.index('Flow')
                fl = np.abs(edge.scalars[flInd])
                flInd = edgeN.scalarNames.index('Flow')
                flN = np.abs(edgeN.scalars[flInd])
                dfl[edgeN.i0:edgeN.i1] = flN - fl
                filledEdge[j] = True

    print(f'Mean dif>0: {np.median(np.abs(dfl)[np.abs(dfl)>0])}')     
    
    i = 0
    if True:   
        cmap = 'bwr'
        cmap_range = [-500,500]
        vis = graphN.plot_graph(edge_color=dfl,cmap=cmap,scalar_color_name='',log_color=False,cmap_range=cmap_range,show=False,block=False,win_width=win_width,win_height=win_height)
        ofile = join(delta_dir,f'delta_flow_{str(i+1).zfill(8)}.png')
        vis.screen_grab(ofile)
        vis.destroy_window()
        
        cmap = 'jet'
        cmap_range = [-3.,12.]
        vis = graphN.plot_graph(cmap=cmap,scalar_color_name='Flow',log_color=True,cmap_range=cmap_range,show=False,block=False,win_width=win_width,win_height=win_height)
        ofile = join(flow_dir,f'log_flow_{str(i+1).zfill(8)}.png')
        vis.screen_grab(ofile)
        vis.destroy_window()
        
        cmap = 'jet'
        cmap_range = [40,70]
        vis = graphN.plot_graph(cmap=cmap,scalar_color_name='Pressure',log_color=True,cmap_range=cmap_range,show=False,block=False,win_width=win_width,win_height=win_height)
        ofile = join(flow_dir,f'pressure_{str(i+1).zfill(8)}.png')
        vis.screen_grab(ofile)
        vis.destroy_window()
        
        graphN.write(join(vessels_dir,'retinal_vein_occlusion.am'))
    print(f'Written {ofile}')

if __name__=='__main__':
    path = '/PATH/TO/SIMULAITON/DATA'
    gfile = join(path,'cco','retina_cco_a2v_reanimate.am')
    geometry_file = join(path,'lsystem','retina_geometry.p')
    
    output_dir = join(path,'retinal_vein_occlusion_movie2')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pfile = None
    retinal_vein_occulsion(gfile=gfile,geometry_file=geometry_file,output_dir=output_dir)        
    
