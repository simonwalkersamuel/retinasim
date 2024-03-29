import os
join = os.path.join
import numpy as np
arr = np.asarray
import pymira.spatialgraph as sp
import open3d as o3d
from retinasim import geometry
from functools import partial
from retinasim.lsystemlib.embedding import embed_graph
from tqdm import tqdm

from retinasim.retina_surface import RetinaSurface
from retinasim.eye import Eye
from retinasim.utility import filter_graph_by_radius

def project_to_surface(graph=None,filename=None,vfile=None,efile=None,mesh_file_format='ply',interpolate=True,filter=True,project=True,plot=False,write_mesh=False,ofile=None,iterp_resolution=5.,eye=None,domain=None):
   
    """ 
    Constrain graph nodes and project onto surface   
    """
    
    if eye is None:
        # Load eye geometry file
        if efile is None:
            efile = join(path,'retina_geometry.p')

        eye = Eye()
        eye.load(efile)
    if domain is None:
        sim_domain = eye.domain
    else:
        sim_domain = domain
        
    # Load surface
    if vfile is None:
        vfile = mesh_ofile.replace(f'.{mesh_file_format}',f'_vessels.{mesh_file_format}')
        
    rs = RetinaSurface(domain=sim_domain,eye=eye)
    mesh = rs.load_surface(vfile)
    vessel_surf = np.asarray(mesh.vertices)
    vessel_surf_normals = np.asarray(mesh.vertex_normals)
    vessel_surf_faces = np.asarray(mesh.triangles)
     
    # Set up open3d and add the blood vessel surface mesh
    scene = o3d.t.geometry.RaycastingScene()  
    mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    # Read graph file
    if graph is None:    
        graph = sp.SpatialGraph()
        graph.read(filename)
        
    vt = graph.get_data('VesselType')
    
    z0 = -105. # IPL
    z1 = -165. # OPL
    z0 = z1
    points = graph.get_data('EdgePointCoordinates').copy()
    nodes = graph.get_data('VertexCoordinates').copy()
    plexus = np.zeros(graph.nnode,dtype='int') - 1
    for e in range(graph.nedge):
        edge = graph.get_edge(e)
        
        estart = graph.get_edges_containing_node(edge.start_node_index)
        vts = []
        for es in estart:
            edge_s = graph.get_edge(es)
            vts.append(edge_s.scalars[1][0])
        vts = arr(vts).astype('int')
        eend = graph.get_edges_containing_node(edge.end_node_index)
        vte = []
        for ee in eend:
            edge_e = graph.get_edge(ee)
            vte.append(edge_e.scalars[1][0])
        vte = arr(vte).astype('int')

        strt_depth, end_depth = 0.,0.        
        if np.all(vts==2) and np.all(vte==2):

            if plexus[edge.start_node_index]==-1:
                plexus[edge.start_node_index] = 1 #np.random.choice([1,2])
                if plexus[edge.start_node_index]==1:
                    nodes[edge.start_node_index,2] = z0
                    strt_depth = z0
                if plexus[edge.start_node_index]==2:
                    nodes[edge.start_node_index,2] = z1
                    strt_depth = z1
            else:
                strt_depth = nodes[edge.start_node_index,2]
                
            if plexus[edge.end_node_index]==-1:
                plexus[edge.end_node_index] = 1 #np.random.choice([1,2])
                if plexus[edge.end_node_index]==1:
                    nodes[edge.end_node_index,2] = z0
                    end_depth = z0
                if plexus[edge.end_node_index]==2:
                    nodes[edge.end_node_index,2] = z1
                    end_depth = z1
            else:
                end_depth = nodes[edge.end_node_index,2]
                
            #points[edge.i0:edge.i1,2] = np.linspace(strt_depth,end_depth,edge.npoints)
        else:
            if np.any(vts==2) and np.any(vts!=2):
                plexus[edge.start_node_index] = 0
                if plexus[edge.end_node_index]<0:
                    plexus[edge.end_node_index] = 1
                    end_depth = z0
                else:
                    end_depth = points[edge.i1-1,2]
                plexus[edge.end_node_index] = 1
                strt_depth = points[edge.i0,2]
                end_depth = z0
                points[edge.i0:edge.i1,2] = np.linspace(strt_depth,end_depth,edge.npoints)
                nodes[edge.end_node_index,2] = end_depth
            if np.any(vte==2) and np.any(vte!=2):
                plexus[edge.end_node_index] = 0
                if plexus[edge.start_node_index]<0:
                    plexus[edge.start_node_index] = 1
                    strt_depth = z0
                else:
                    strt_depth = points[edge.i0,2]
                end_depth = points[edge.i1-1,2]
                #points[edge.i0:edge.i1,2] = np.linspace(strt_depth,end_depth,edge.npoints)
                nodes[edge.start_node_index,2] = strt_depth
                
    for e in range(graph.nedge):
        edge = graph.get_edge(e)
        if plexus[edge.start_node_index]>0 or plexus[edge.end_node_index]>0:
            strt_depth = nodes[edge.start_node_index,2]
            end_depth = nodes[edge.end_node_index,2]
            points[edge.i0:edge.i1,2] = np.linspace(strt_depth,end_depth,edge.npoints)
            
    graph.set_data(nodes,name='VertexCoordinates')
    graph.set_data(points,name='EdgePointCoordinates')
    
    #ofile = filename.replace('.am','_plexi.am')
    #graph.write(ofile)
    
    # Filter vessel radii
    if filter:
        min_filter_radius = 5.
        filter_clip_radius = 0.
        filter_graph_by_radius(graph,min_filter_radius=min_filter_radius,filter_clip_radius=filter_clip_radius,write=False)
    
    if ofile is None:
        ofile = filename.replace('.am','_projected.am')

    # Interpolate line segments
    if interpolate:
        print('Interpolating...')

        interp_mode = 'linear'
        if interp_mode=='linear':
            ed = sp.Editor()
            ed.interpolate_edges(graph,interp_resolution=iterp_resolution,filter=None,noise_sd=0.)
        else:
            coords = graph.get_data('VertexCoordinates')
            points = graph.get_data('EdgePointCoordinates')
            npoints = graph.get_data('NumEdgePoints')
            conns = graph.get_data('EdgeConnectivity')
            radii = graph.get_data('Radii')
            vType = graph.get_data('VesselType')
            midLinePos = graph.get_data('midLinePos')
            if midLinePos is None:
                midLinePos = np.zeros(radii.shape[0],dtype='int')
            from retinasim.simplex_path import simplex_path,simplex_path_multi
            ninterp = 20
            interp_points = np.zeros([(points.shape[0]-1)*ninterp,3])

            #new_coords = coords.copy()
            
            start_dir,end_dir,fade,weight_mode = None,None,0.2,'linear'
            feature_size = [np.mean(diameters)*10.]
            displacement = 0.05 # 0.1=very curvy
            new_points[:,0:2] = simplex_path_multi(points[:,0:2],start_dir=start_dir,end_dir=end_dir,fade=fade,feature_size=[feature_size],npoints=ninterp,displacement=displacement,weight_mode=weight_mode) 

    coords = graph.get_data('VertexCoordinates')
    coords_snap = coords.copy()
    points = graph.get_data('EdgePointCoordinates')
    points_snap = points.copy()
    
    coords_or = coords.copy()
    points_or = points.copy()

    if project: # Project vessel data onto sphere
        
        domain_size = sim_domain[:,1] - sim_domain[:,0]
        
        print('Projecting vertices onto sphere...')
        for j,crd in enumerate(tqdm(coords)):
            # Draw vector from centre of the eye to coords
            surface_height = crd[2] #0.
            P = eye.occular_centre - arr([crd[0],crd[1],surface_height])
            # Rotation at retina surface
            rt = geometry.align_vectors(arr([0.,0.,1.]),P)
            # Translate point to surface, rotate, then translate back
            coords_snap[j] = np.dot(rt,crd)
            
        print('Projecting edgepoints onto sphere...')
        for j,crd in enumerate(tqdm(points)): 
            # Draw vector from centre of the eye to coords
            surface_height = crd[2] #0.
            P = eye.occular_centre - arr([crd[0],crd[1],surface_height])
            # Rotation at retina surface
            rt = geometry.align_vectors(arr([0.,0.,1.]),P)
            # Translate point to surface, rotate, then translate back
            points_snap[j] = np.dot(rt,crd)
        coords = coords_snap
        points = points_snap
        
        # Snap to mesh for fine adjustment after sphere projection
        snap = True
        if snap:   
            # Ray tracing 
            print('Ray tracing for surface embedding...')
            
            def rt_snap(crds,centre,displacement=None):
                xg = crds[:,0].flatten()
                yg = crds[:,1].flatten()
                zg = crds[:,2].flatten()

                dir_ = np.reshape(centre,(3,1))-np.vstack([xg,yg,zg])
                dirn_ = dir_.copy()
                dist = np.linalg.norm(dir_,axis=0)
                dirn_ /= dist
                dir_ = dir_.transpose()
                dirn_ = dirn_.transpose()
                    
                ray_dirs = np.vstack([np.zeros(crds.shape[0])+centre[0],np.zeros(crds.shape[0])+centre[1],np.zeros(crds.shape[0])+centre[2],-dirn_[:,0],-dirn_[:,1],-dirn_[:,2]]).transpose()

                # Convert to tensors
                rays = o3d.core.Tensor(ray_dirs,dtype=o3d.core.Dtype.Float32)
                ans = scene.cast_rays(rays)
                t_hit = ans['t_hit'].numpy()
                
                coords_snap = centre - dirn_*np.reshape(t_hit,[t_hit.shape[0],1]) + (displacement*dirn_.transpose()).transpose()
                return coords_snap
            
            coords_snap = rt_snap(coords,eye.occular_centre,displacement=coords_or[:,2])
            points_snap = rt_snap(points,eye.occular_centre,displacement=points_or[:,2])
    
    graph.set_data(coords_snap,name='VertexCoordinates')
    graph.set_data(points_snap,name='EdgePointCoordinates')
    
    if plot:
        gmesh = graph.plot_graph(plot=False) #,min_radius=20.)
        breakpoint()
        o3d.visualization.draw_geometries([gmesh,mesh],mesh_show_back_face=True,mesh_show_wireframe=False) 

    if ofile!='':
        print(f'Writing graph to {ofile}')
        graph.write(ofile)
    
    if write_mesh:
        print('Writing vessel mesh...')
        vtypeEdge = graph.point_scalars_to_edge_scalars(name='VesselType')
        tp = graph.plot_graph(show=False,block=False,min_radius=5.,edge_filter=vtypeEdge==0,cyl_res=10,radius_scale=1)
        gmesh_artery = tp.cylinders_combined
        ofile2 = ofile.replace('.am','_artery.ply')
        o3d.io.write_triangle_mesh(ofile2,gmesh_artery)
        tp.destroy_window()
        tp = graph.plot_graph(show=False,block=False,min_radius=5.,edge_filter=vtypeEdge==1,cyl_res=10,radius_scale=1)
        gmesh_vein = tp.cylinders_combined
        ofile2 = ofile.replace('.am','_vein.ply')
        o3d.io.write_triangle_mesh(ofile2,gmesh_vein)
        
        #if True: # Switch off if capillaries not required as separate mesh
        #    tp = graph.plot_graph(show=False,block=False,min_radius=5.,edge_filter=vtypeEdge==2,cyl_res=5,radius_scale=1)
        #    gmesh_cap = tp.cylinders_combined
        #    ofile2 = ofile.replace('.am','_capillary2.ply')
        #    o3d.io.write_triangle_mesh(ofile2,gmesh_cap)
        
        tp.destroy_window()
        
    return graph
