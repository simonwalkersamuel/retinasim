from pymira import spatialgraph
from retinasim.utility import reanimate_sim
from retinasim.utility import join_feeding_vessels, make_dir, remove_endpoint_nodes, remove_all_endpoints, identify_inlet_outlet
from retinasim.capillary_bed import direct_arterial_venous_connection, voronoi_capillary_bed, remove_arteriole
from retinasim.cco import retina_lsystem, retina_inject, retina_vascular
from retinasim.scripts.fill_avascular_region import fill_avascular_region
from retinasim import geometry
from retinasim.eye import Eye

import numpy as np
arr = np.asarray
import os
join = os.path.join
from tqdm import trange, tqdm

def dbr_sim(dataPath=None,gfile=None,opath=None,ofile=None,nart=500,eye=None,graph=None,max_radius=35.,min_radius=0.,\
            min_displacement=8000.,win_width=6000,win_height=6000,plot=True,fluorescein=True,write_graph=True,grabfile='', \
            a_pressure=80.,v_pressure=30.,catId=0):

    """
    Simulate diabetic retinopathy
    """

    if eye is None:
        geometry_file = join(dataPath,"lsystem","retina_geometry.p")
        eye = Eye()
        eye.load(geometry_file)
    
    if graph is None:
        graph = spatialgraph.SpatialGraph()
        graph.read(join(dataPath,"cco",gfile))
    
    if opath is not None:    
        make_dir(opath)
        concPath = join(opath,'concentration')
        make_dir(concPath)
    else:
        concPath = None

    if catId==0:
        vtype = 'arteriole'
    elif catId==1:
        vtype = 'venule'
    print(f'Removing {nart} random {vtype} points...')
    rem_edge = remove_arteriole(graph,catId=catId,n=nart,max_radius=max_radius,min_radius=min_radius,grab_file=grabfile,eye=eye,min_displacement=min_displacement,remove_endpoints=False,select_only=True)  

    rem_pts = []
    for re in rem_edge:
        edge = graph.get_edge(re)
        rem_pts.append(np.mean(edge.coordinates,axis=0))
    rem_pts = arr(rem_pts)

    # Get flow ordering
    graph = retina_inject.crawl(graph,proj_path=opath,calculate_conc=False,image_dir=concPath,plot=plot,ghost_edges=rem_edge)
    flow = graph.get_data('Flow')
    perf = graph.get_data('Perfused')
    
    if write_graph:
        graph.write(join(opath,'retina_dbr.am'))

    epi = graph.edgepoint_edge_indices()
    rind = np.where(perf==0)
    redge = np.unique(epi[rind])
    gvars = spatialgraph.GVars(graph)
    gvars.remove_edges(redge)
    graph = gvars.set_in_graph()
    
    inlets,outlets = identify_inlet_outlet(graph)
    graph,ep_nodes = remove_all_endpoints(graph,keep=np.concatenate([[inlets],[outlets]]),return_nodes=True)

    breakpoint()
    avregs,avmeshes = [],[]
    rem_pts = np.vstack([rem_pts,ep_nodes])
    
    print(f'***Setting {rem_pts.shape[0]} random walkers***')
    
    breakpoint()
    polygons = []
    for re in tqdm(rem_pts):
        # Check if point is already contained inside an existing polygon...
        from shapely import geometry as sgeometry        
        contained = False
        for polygon in polygons:
            if polygon.contains(sgeometry.Point(re[:2])):
                contained = True
                break
        
        if contained==False:
            avreg,mesh = fill_avascular_region(graph, re, nsteps=50)
            if avreg is not None:
                avregs.append(avreg)
                avmeshes.append(mesh)
                tris = np.asarray(mesh.triangles)
                tris = np.hstack([tris,tris[:,0][...,None]])
                verts = np.asarray(mesh.vertices)
                polygons.extend([sgeometry.Polygon(sgeometry.LineString(verts[t,:2])) for t in tris])
        
    avmeshes = np.sum(avmeshes)
    
    ofile = 'retina_dbr_reanimate.am'
    graph = reanimate_sim(graph,opath=opath,ofile=ofile,a_pressure=a_pressure,v_pressure=v_pressure)
    
    if plot==True:
        breakpoint()

        vt = graph.point_scalars_to_edge_scalars(name='VesselType')
        edge_filter = np.ones(vt.shape[0])
        edge_filter[vt==2] = False
        vis = graph.plot_graph(bgcolor=[1.,1.,1.],edge_filter=edge_filter,scalar_color_name='VesselType',show=False,block=False,win_width=win_width,win_height=win_height,additional_meshes=avmeshes)

        vis.screen_grab(join(opath,'artery_vein_dbr.png'))
        vis.set_cylinder_colors(scalar_color_name='Flow',cmap='jet',log_color=True,cmap_range=[None,None])
        vis.screen_grab(join(opath,'log_flow.png'))
        vis.set_cylinder_colors(scalar_color_name='Pressure',cmap='jet',log_color=False,cmap_range=[v_pressure,a_pressure])
        vis.screen_grab(join(opath,'pressure.png'))
        vis.destroy_window()
    
    # Virtual injection of fluorescein
    if fluorescein==True:
        graph = retina_inject.crawl(graph,proj_path=opath,calculate_conc=True,image_dir=concPath)
        pfile = join(concPath,"concentration_data.p")
        with open(pfile,'rb') as handle:
            data = pickle.load(handle)
            conc,t = data
        retina_inject.conc_movie(graph,data=conc,time=t,recon_times=recon_times,odir=concPath,win_width=win_width,win_height=win_width,eye=eye,movie_naming=False)    
    
    return graph, rem_edge, avmeshes
    
def dbr_iterative(niteration=20,dataPath=None,gfile=None,opath=None,ofile=None,eye=None,graph=None):

    """
    Simulate diabetic retinopathy, iteratively
    """

    if eye is None:
        geometry_file = join(dataPath,"lsystem","retina_geometry.p")
        eye = Eye()
        eye.load(geometry_file)
    
    if graph is None:
        graph = spatialgraph.SpatialGraph()
        graph.read(join(dataPath,"cco",gfile))
        
    min_displacement = np.linspace(10000.,1000.,niteration)
    sample_area = np.pi*np.square(min_displacement)
    nart = (np.concatenate([[1.],1+(sample_area[:-1]-sample_area[1:])/sample_area[1:]]) * 50).astype('int')
    
    #nart = np.linspace(50,100,niteration,dtype='int')
    
    for i in range(niteration):
        opath_cur = join(opath,f'dbr_{str(i).zfill(8)}')
        make_dir(opath_cur)
        grabfile = join(opath_cur,'artery_vein_artrem.png')
        graph,_ = dbr_sim(eye=eye,graph=graph,nart=nart[i],opath=opath_cur,max_radius=35.,min_displacement=min_displacement[i],grabfile=grabfile)
