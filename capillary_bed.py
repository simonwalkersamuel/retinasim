from pymira import spatialgraph
import numpy as np
import os
join = os.path.join
arr = np.asarray
from tqdm import trange, tqdm
from matplotlib import pyplot as plt
from shapely.geometry import LineString, MultiLineString
import open3d as o3d
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import qmc

from pymira.spatialgraph import update_array_index,delete_vertices,GVars
from retinasim import geometry
from retinasim.eye import Eye
from retinasim.utility import test_midline_collision, set_and_write_graph, get_graph_fields, filter_graph_by_radius, interpolate_graph, get_node_count, remove_endpoint_nodes, create_edge_node_lookup, remove_all_endpoints, identify_inlet_outlet

def inside_vol(v0,extent,margin=arr([0.,0.,0.]),centre=arr([0.,0.]),radius=35000.,shape='circ'):
  
    if shape=='rect':
        sind = np.where( (np.all(v0>=extent[:,0]+margin,axis=1)) & (np.all(v0<=extent[:,1]-margin,axis=1)) )
        if len(v0.shape)==1:
            if len(sind[0])>0:
                return True
        else:
            res = np.zeros(v0.shape[0],dtype='bool')
            res[sind[0]] = True
            return res
    elif shape=='circ':
        if len(v0.shape)==1:
            dist = np.linalg.norm(v0-centre)
            if dist<=radius:
                return True
        else:
            dist = np.linalg.norm(v0-centre,axis=1)
            return dist<=radius
            
def halton_capillary(npoints=10000,radius=0.5,feature_radius=0.2,centre=arr([0.5,0.5]),A=300.,B=1.0,plot=False):

    sampler = qmc.Halton(d=2, scramble=True)

    pts = np.zeros([npoints,2])
    pts_filled = np.zeros(npoints,dtype='bool')

    i = 0
    domain_centre = arr([0.5,0.5])
    pbar = tqdm(total=npoints)
    while True:
        pt = sampler.random(n=1)
        dist = np.linalg.norm(pt-domain_centre)
        feature_dist = np.linalg.norm(pt-centre)
        if dist<radius:
            # If points are within the feature, impose exponential selection
            if feature_dist<=feature_radius and np.random.uniform()>B*np.exp(-feature_dist/A): #0.25:
                pts_filled[i] = True
                pts[i] = pt
            elif feature_dist>feature_radius:
                pts_filled[i] = True
                
            if pts_filled[i]:
                pts[i] = pt
                i += 1
                
                pbar.update(1)
            
            if i>=npoints:
                break
                
    pbar.close()

    if plot:
        vor = Voronoi(pts[:,0:2],qhull_options='Qbb Qc Qx')
        fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',line_width=2, line_alpha=0.6, point_size=1, figsize=[1.5,1.5])
        plt.show()
        
    return pts            
        
def plot_voronoi(vor_verts,simplexes,graph=None,meshes=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vor_verts)
    pcd.paint_uniform_color([0, 0, 1])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vor_verts)
    line_set.lines = o3d.utility.Vector2iVector(simplexes)
    m = [pcd,line_set]
    if meshes is not None:
        m.extend(meshes)
        
    if graph is not None:
        vis = graph.plot_graph(show=False,block=False)
        cyls = vis.cylinders_combined
        ms = [line_set,cyls]
        o3d.visualization.draw_geometries(ms,mesh_show_wireframe=True) 
    else:   
        o3d.visualization.draw_geometries(m,mesh_show_wireframe=True) 

def find_voronoi_graph_intersections(edge_region,vertices,coordinates,edge_point_index,region_index,region_simplexes,region_simplex_inds):

    """
    Find intersections between Voronoi region edges and the current graph edge
    edge_region: vertex indices for the Voronoi edge
    vertices: coordinates for each vertex in the Voronoi edge
    coordinates: edgepoint coordiantes for the current graph edge
    edge_point_index: which graph edge (index) each the edgepoints come from 
    region_index: which region (index) the Voronoi edges come from
    region_simplexes: Simplexes corresponding to each Voronoi edge
    region_simplex_inds: Indices of simplexes corresponding to each Voronoi edge
    """

    intersections,intersecting_simplex,non_intersecting = [],[],[]

    # Complete the loop
    edge_region = arr([edge_region+[edge_region[0]]]).squeeze()
    vertices = np.vstack([vertices,vertices[0]])

    if np.all(edge_region>0):
        # Loop through each edge of the voronoi edges an look for intersections with the graph edge
        for j in range(1,edge_region.shape[0]):
            # Candidate Voronoi edges to test for intersection with graph
            v0 = np.concatenate([vertices[j-1],[0.]]).astype('float32')
            v1 = np.concatenate([vertices[j],[0.]]).astype('float32')
            e0s = coordinates[:-1]
            e1s = coordinates[1:] #np.roll(edge.coordinates,-1,axis=0)[1:]
            # Create a copy of the edge with the third element (z) set to 0
            e0s_0 = e0s.copy()
            e1s_0 = e1s.copy()
            e0s_0[:,2] = 0.
            e1s_0[:,2] = 0.
            
            # Test for collision between current simplex and all segments of current edge
            test = geometry.segment_intersection_multi(v0,v1,e0s_0.transpose(),e1s_0.transpose())
            
            if np.any(test):
                e0ints,e1ints = e0s[test],e1s[test]
                for k,e0 in enumerate(e0ints):
                    _,int_point = geometry.segments_intersect_2d(v0[0:2],v1[0:2],e0ints[k][0:2],e1ints[k][0:2])

                    int_point_3d = np.append(int_point,np.mean([e0ints[k][2],e1ints[k][2]]))
                    intersections.append({'edge':edge_point_index,'intersection':int_point_3d,'region':region_index,'vertices':[edge_region[j-1],edge_region[j]]})
                    
                    # Get all the simplex indices for the current region
                    if len(region_simplex_inds)>0:
                        edge_simp_inds = region_simplex_inds #[j]
                        # Get the simplexes they correspond to
                        edge_simps = region_simplexes #[j] # simplexes[edge_simp_inds]
                        if len(edge_simps.shape)==1:
                            edge_simps = np.reshape(edge_simps,[1,edge_simps.shape[0]])
                        # Find which simplex has been intersected
                        ints = intersections[-1]
                        #print(edge_simps,ints)
                        
                        sind = np.where( ((edge_simps[:,0]==ints['vertices'][0]) & (edge_simps[:,1]==ints['vertices'][1])) | ((edge_simps[:,0]==ints['vertices'][1]) & (edge_simps[:,1]==ints['vertices'][0])))
                        if len(sind[0])>0:
                            if len(edge_simp_inds.shape)>0:
                                intersecting_simplex.append(edge_simp_inds[sind[0]])
                            else:
                                if sind[0]==0:
                                    intersecting_simplex.append(edge_simp_inds)
                                else:
                                    breakpoint()      
                    
    return intersections, intersecting_simplex
    
def find_voronoi_graph_intersections_multi(edge_region_multi,vertices_multi,coordinates_multi,edge_point_index_multi,region_inds,region_simplexes,region_simplex_inds,proc_ind=0):

    """
    Find intersections between Voronoi edges and graph edges
    edge_region: vertex indices for the Voronoi edge
    vertices: coordinates for each vertex in the Voronoi edge
    coordinates: edgepoint coordiantes for the current graph edge
    edge_point_index: which graph edge (index) each the edgepoints come from 
    region_index: which region (index) the Voronoi edges come from
    region_simplexes: Simplexes corresponding to each Voronoi edge
    region_simplex_inds: Indices of simplexes corresponding to each Voronoi edge
    """

    nreg = len(edge_region_multi)
    intersections, intersecting_simplex = [],[]
    #print(f'PROC, {nreg}')
    
    if proc_ind==0:
        rng_func = trange
    else:
        rng_func = range
    
    for i in rng_func(nreg):
        ints,intsimps = find_voronoi_graph_intersections(edge_region_multi[i],vertices_multi[i],coordinates_multi[i],edge_point_index_multi[i],region_inds[i],region_simplexes[i],region_simplex_inds[i])
        intersections.extend(ints)
        intersecting_simplex.extend(intsimps)
        
    return intersections,intersecting_simplex
    
def create_edges_from_intersections(edges_intersected,intersections,start_node_index_multi,start_node_coords_multi,end_node_index_multi,end_node_coords_multi,scalars_multi,max_connect_radius=30.,radius_index=0):

    remove_edge_inds = []
    # Loop through all intersected edges
    
    new_edges = []
    new_nodes = []
    remove_edge_inds = []
    
    for e,eind in enumerate(edges_intersected):
        cur_intersections = [x for x in intersections if x['edge']==eind]
        cur_pts = arr([x['intersection'] for x in cur_intersections])

        edge_radii = scalars_multi[e][radius_index] #edge.scalars[edge.scalarNames==gvars.radname]
        
        if edge_radii<=max_connect_radius:
            start_node = start_node_coords_multi[i] # edge.start_node_coords
            end_node = end_node_coords_multi[i] # edge.end_node_coords
            scvals = [x[0] for x in scalars_multi[e]]
            
            # Ordering of new points
            dists = np.linalg.norm(cur_pts-start_node,axis=1)
            srt = np.argsort(dists)
            cur_intersections = [cur_intersections[i] for i in srt]
            cur_pts = cur_pts[srt]
            
            count = 0
            for i,pt in enumerate(cur_pts):
            
                # Check for degeneracy. If not, carry on...
                sind = np.where((gvars.nodecoords[:,0]==pt[0]) & (gvars.nodecoords[:,1]==pt[1]) & (gvars.nodecoords[:,2]==pt[2]))
                if len(sind[0])==0:
                
                    # Create a node
                    #gvars.add_node(pt)
                    new_nodes.append(pt)
              
                    # Replicate edge
                    if count==0:
                        # Connect new node to start node
                        scvals[radius_index] = edge_radii[0]
                        #gvars.add_edge(edge.start_node_index, gvars.node_ptr-1, scvals)
                        new_edge.append([start_node_index_multi[e],len(new_nodes)-1,scvals])
                    else:
                        # Connect new node to previous node
                        scvals[radius_index] = edge_radii[0]
                        #gvars.add_edge(gvars.node_ptr-2, gvars.node_ptr-1, scvals)
                        new_edge.append([len(new_nodes)-2,len(new_nodes)-1,scvals])

                    # Connect to Voronoi
                    scvals[radius_index] = 3.
                    if np.random.uniform()>0.5:
                        #gvars.add_edge(gvars.node_ptr-1, cur_intersections[i]['vertices'][0]+gvars.vnode_0, scvals)
                        new_edge.append([len(new_nodes)-1,cur_intersections[i]['vertices'][0],vnode_0,scvals])
                    else:
                        #gvars.add_edge(gvars.node_ptr-1, cur_intersections[i]['vertices'][1]+gvars.vnode_0, scvals)
                        new_edge.append([len(new_nodes)-1,cur_intersections[i]['vertices'][0],vnode_0,scvals])
                    count += 1

            # Add final connection point at end
            scvals[radius_index] = edge_radii[0]
            #gvars.add_edge(gvars.node_ptr-1, edge.end_node_index, scvals)
            new_edge.append([len(new_nodes)-1,end_node_index_multi[e],scvals])
            
            remove_edge_inds.append(eind)
            
def voronoi_points(graph=None,regular_mesh=False,extent=None,spacing=150,eye=None,radial_extent=None,radial_centre=None):

    if regular_mesh:
        xv,yv = np.meshgrid(np.arange(extent[0,0],extent[0,1],spacing),np.arange(extent[1,0],extent[1,1],spacing))
        #zv = np.zeros(npts[2])
        grid_points = np.vstack([xv.flatten(),yv.flatten()]).transpose()
        # Convert to 3D
        grid_points = np.hstack([grid_points,np.zeros([grid_points.shape[0],1])]).astype('float32')
    else:
        vol_size = extent[:,1]-extent[:,0]
        linear_density = np.round(vol_size/spacing)
        npoints = int(linear_density[0]*linear_density[1])
        print(f'Generated {npoints} Voronoi points (spacing {spacing}um, density {linear_density})')
        
        if False:
            from scipy.stats import qmc
            sampler = qmc.Halton(d=2, scramble=True)
            grid_points = sampler.random(n=npoints)
        else:
            feature_radius = eye.fovea_radius*5. / (eye.domain[0,1]-eye.domain[0,0])
            centre = (eye.macula_centre / ((eye.domain[:,1]-eye.domain[:,0]))) + 0.5
            grid_points = halton_capillary(npoints=npoints,feature_radius=feature_radius,centre=centre[:2],A=feature_radius,B=1.,plot=False)  # 0.25*feature_radius          
            
        # Scale to volumes
        grid_points[:,0] = grid_points[:,0]*vol_size[0] + extent[0,0]
        grid_points[:,1] = grid_points[:,1]*vol_size[1] + extent[1,0]
        # Convert to 3D
        grid_points = np.hstack([grid_points,np.zeros([grid_points.shape[0],1])]).astype('float32')

        #edgeInds = graph.edgepoint_edge_indices()
        #samp_dist = 1000.
        remove = np.zeros(grid_points.shape[0],dtype='bool')
        
        #breakpoint()
        if radial_extent is not None:
            dist = np.linalg.norm(grid_points-radial_centre,axis=1)
            remove[dist>radial_extent] = True

        grid_points = grid_points[~remove]
    
        return grid_points
    
def create_capillary_voronoi(graph,eye=None,regular_mesh=False,voronoi_vessel_avoidance=True,avoid_midline=True,avoid_macula=True,avoid_optic_disc=False,spacing=150.,simplex_graph_collision_resolution=True,remove_voronoi_nodes_inside_vessels=True,write=True,ofile=None,capillary_radius=3.,max_connection_vessel_radius=20.,radial_extent=None,radial_centre=None):
    print('Calculating voronoi')

    # Identify arterial inlets (2) and venous outlets (2)
    inlet1,outlet1 = identify_inlet_outlet(graph)
    inlet2,outlet2 = identify_inlet_outlet(graph,ignore=[inlet1,outlet1])
    if inlet2 is not None:
        inlets = [inlet1,inlet2]
    else:
        inlets = [inlet1]
    if outlet2 is not None:
        outlets = [outlet1,outlet2]
    else:
        outlets = [outlet1]

    # Create lightweight object for editing graph            
    gvars = GVars(graph)
    extent = np.asarray(graph.edge_spatial_extent())

    # Create Voronoi points
    grid_points = voronoi_points(graph=graph,regular_mesh=False,extent=extent,spacing=spacing,eye=eye,radial_extent=radial_extent,radial_centre=radial_centre)

    # Create an array linking each edgepoint to the edge index it belongs to
    edgepointInds = graph.edgepoint_edge_indices()

    # Combine grid points with graph edgepoints
    concat_points = True # TESTING
    endpoints_only = True
    # Identify node branch number
    gc = graph.get_node_count()
    if concat_points:
        edgepoints_filt = gvars.edgepoints
        edgepoint_lookup = np.zeros(edgepoints_filt.shape[0],dtype='int')-1
        #breakpoint()
        if radial_extent is not None:
            dist = np.linalg.norm(edgepoints_filt-radial_centre,axis=1)
            edgepoint_lookup = np.where(dist<=radial_extent)[0]
            edgepoints_filt = edgepoints_filt[dist<=radial_extent]
        else:
            edgepoint_lookup = np.linspace(0,edgepoints_filt.shape[0]-1,edgepoints_filt.shape[0])
        if endpoints_only:
            stmp = np.where(gc==1)
            end_edgepoint_inds = np.zeros(stmp[0].shape[0],dtype='int') - 1
            for j,si in enumerate(tqdm(stmp[0])):
                edgeInd = graph.get_edges_containing_node(si)
                edge = graph.get_edge(edgeInd[0])
                if edge.start_node_index==si and si in edgepoint_lookup:
                    end_edgepoint_inds[j] = edge.i0
                elif edge.end_node_index==si and si in edgepoint_lookup:
                    end_edgepoint_inds[j] = edge.i1-1
                else:
                    pass
                    
            end_edgepoint_inds = end_edgepoint_inds[end_edgepoint_inds>=0]
            edgepoints_filt = gvars.edgepoints[end_edgepoint_inds]
            edgepoint_lookup = end_edgepoint_inds
            
        points = np.concatenate([grid_points,edgepoints_filt])
        # Keep track of where each of the points came from
        point_type = np.ones(points.shape[0],dtype='int')
        point_type[:grid_points.shape[0]] = 0
        point_edgepoint_origin = np.zeros(points.shape[0],dtype='int') - 1
        point_edgepoint_origin[grid_points.shape[0]:] = edgepoint_lookup
    else:
        # Keep track of where each of the points came from
        edgepoints_filt = gvars.edgepoints
        edgepoint_lookup = np.zeros(edgepoints_filt.shape[0],dtype='int')-1
        #breakpoint()
        if radial_extent is not None:
            dist = np.linalg.norm(edgepoints_filt-radial_centre,axis=1)
            edgepoint_lookup = np.where(dist<=radial_extent)[0]
            edgepoints_filt = edgepoints_filt[dist<=radial_extent]
        else:
            edgepoint_lookup = np.linspace(0,edgepoints_filt.shape[0]-1,edgepoints_filt.shape[0])
        point_type = np.ones(gvars.edgepoints.shape[0],dtype='int')
        points = grid_points
        point_edgepoint_origin = np.zeros(points.shape[0],dtype='int') - 1
        #point_edgepoint_origin[grid_points.shape[0]:] = edgepoint_lookup
    npoints = points.shape[0]

    #breakpoint()
    vor = Voronoi(points[:,0:2],qhull_options='Qbb Qc Qx')
    
    vor_verts = vor.vertices
    simplexes = arr(vor.ridge_vertices)
    
    regions = [vor.regions[vor.point_region[i]] for i in range(npoints)] 
    if vor.point_region.shape[0]!=npoints:
        print('Voronoi error!')
        breakpoint()
        
    # Mark which region corresponds to an edgepoint
    edge_regions = [x for i,x in enumerate(regions) if point_type[i]==1]
    edge_point_inds = np.where(point_type==1)[0]
    vor_regions = [x for i,x in enumerate(regions) if point_type[i]==0]
        
    ### Locate intersections between graph edges and voronoi simplices ###
    nreg = len(edge_regions)
    print('Mapping simplex edges to regions...')
    verts_used = np.unique(vor.ridge_points)
    srt0 = np.argsort(vor.ridge_points[:,0])
    bc0 = np.bincount(vor.ridge_points[:,0][srt0])
    srt1 = np.argsort(vor.ridge_points[:,1])
    bc1 = np.bincount(vor.ridge_points[:,1][srt1])
    # Make equal length
    if bc0.shape[0]<npoints: #<bc1.shape[0]:
       dif = npoints-bc0.shape[0] #bc1.shape[0]-bc0.shape[0]
       bc0 = np.append(bc0,np.zeros(dif,dtype='int'))
    if bc1.shape[0]<npoints: #bc0.shape[0]:
       dif = npoints-bc1.shape[0] # bc0.shape[0]-bc1.shape[0]
       bc1 = np.append(bc1,np.zeros(dif,dtype='int'))
    
    # Find simplexes bordering each point
    point_simplex_inds = [[] for i in range(npoints)]
    for pnt_ind in trange(npoints):
        #pnt_ind = edge_point_inds[reg_ind]
        i0 = np.sum(bc0[:pnt_ind])
        i1 = i0 + bc0[pnt_ind]
        simp_inds_0 = srt0[i0:i1]
        test0 = np.all(vor.ridge_points[:,0][srt0[i0:i1]]==pnt_ind)
        
        i0 = np.sum(bc1[:pnt_ind])
        i1 = i0 + bc1[pnt_ind]
        simp_inds_1 = srt1[i0:i1]
        test1 = np.all(vor.ridge_points[:,1][srt1[i0:i1]]==pnt_ind)
        
        point_simplex_inds[pnt_ind] = np.concatenate([simp_inds_0,simp_inds_1])
        
        #if len(point_simplex_inds[pnt_ind])==0:
        #    breakpoint()
        
        if not test1 or not test0:
            breakpoint()
    
    # Create endpoint connections
    send = np.where(gc==1)   
    endpoint_conn = []
    print('Calculating endpoint connections...')
    # Loop through each endpoint
    remove_simps = np.zeros(simplexes.shape[0],dtype='bool')
    for si in tqdm(send[0]):
        edgeInd = graph.get_edges_containing_node(si)
        edge = graph.get_edge(edgeInd[0])
        if ( endpoints_only==False and edge.i0 in edgepoint_lookup and edge.i1-1 in edgepoint_lookup ) or ( endpoints_only==True and (edge.i0 in edgepoint_lookup or edge.i1-1 in edgepoint_lookup) ):
            if edge.start_node_index==si:
                stmp = np.where(edgepoint_lookup==edge.i0)[0]
                er = edge_regions[stmp[0]]
                pt1 = edge.start_node_coords
                pt0 = edge.coordinates[1]
                dr = pt1 - pt0
                simp_inds = point_simplex_inds[edge_point_inds[stmp[0]]]
                simps = simplexes[simp_inds]
                reverse = True
            elif edge.end_node_index==si:
                stmp = np.where(edgepoint_lookup==edge.i1-1)[0]
                er = edge_regions[stmp[0]]
                pt1 = edge.end_node_coords
                pt0 = edge.coordinates[-2]
                dr = pt1 - pt0
                simp_inds = point_simplex_inds[edge_point_inds[stmp[0]-1]]
                simps = simplexes[simp_inds]
                reverse = False
            else:
                breakpoint()
                
            # Find closest vertex
            edge_region = arr([er+[er[0]]]).squeeze()
            if np.all(edge_region>0):
                vor_reg_verts = vor.vertices[edge_region]
                dists = np.linalg.norm(vor_reg_verts-pt1[:2],axis=1)
                angles = geometry.vector_angle(dr[:2],vor_reg_verts-pt1[:2])
                if np.any(dists==0) or np.linalg.norm(dr)==0.:
                    breakpoint()
                    
                arg = np.dot(np.reshape(dr[:2],[1,2]),(vor_reg_verts-pt1[:2]).transpose()).squeeze() #/ (dists*np.linalg.norm(dr))
                arg = np.clip(arg,-1,1)
                angles = np.arccos(arg)
                conn_ind = np.argmin(np.abs(angles))
                    
                #angles = np.arccos(np.dot(np.reshape(dr[:2],[1,2]),(vor_reg_verts-pt1[:2]).transpose()).squeeze() / (dists*np.linalg.norm(dr)))
                #conn_ind = np.argmin(angles)
                
                # To avid hyper-connected nodes, if more than 2 connections identified, remove one
                stmp = np.where((simps[:,0]==edge_region[conn_ind]) | (simps[:,1]==edge_region[conn_ind]))
                if len(stmp[0])>2:
                    simpi = simp_inds[stmp]
                    remove_simps[simpi[np.random.choice(stmp[0])]] = True
                endpoint_conn.append({'vertex':edge_region[conn_ind],'node':si,'edge_index':edgeInd[0]})
                
                if False:
                    plt.plot(vor_reg_verts[:,0],vor_reg_verts[:,1])
                    plt.plot([pt0[0],pt1[0]],[pt0[1],pt1[1]])
                    plt.scatter(vor_reg_verts[conn_ind,0],vor_reg_verts[conn_ind,1])
                    plt.scatter(pt1[0],pt1[1])
                    plt.show()
                    breakpoint()
    
    #breakpoint()
    if False: # TEST
        rind = 0 # region index
        simpInds = arr(point_simplex_inds[rind])
        simps = simplexes[simpInds]
        sverts = vor.vertices[simps]
        for i in range(sverts.shape[0]): plt.scatter(sverts[i,:,0],sverts[i,:,1])
        
        # Get point associated with the region
        er = arr(regions[rind])
        everts = vor.vertices[er]
        for i in range(everts.shape[0]): plt.scatter(everts[:,0],everts[:,1])
        plt.show()
     
    intersections = []
    intersecting_simplex = []
    find_intersections = False
    if find_intersections:
        print('Finding intersections...')
        # Parallelise? -----------------------------
        parallel = False
        edgepoints = graph.get_data('EdgePointCoordinates')
        nedgepoints = graph.get_data('NumEdgePoints')
        if find_intersections and parallel:
            if True:
                from multiprocessing import Pool
                nproc = 2
                pool = Pool(nproc)
                nreg = len(edge_regions)
                n_reg_per_proc = int(np.ceil(nreg / float(nproc)))
                
                print(f"{nreg} regions in Voronoi to process. {n_reg_per_proc} regions per process ({nproc} processes)")
                
                args = [[] for i in range(nproc)]
                ptr = 0
                print('Constructing parallel processes...')
                i = 0
                
                breakpoint()
                for i0 in trange(0,nreg,n_reg_per_proc):
                    #i0,i1 = ptr,np.clip(ptr+n_reg_per_proc,0,nreg)
                    i1 = np.clip(i0 + n_reg_per_proc,0,nreg)
                    # Region indices
                    inds = edge_point_inds[np.arange(i0,i1-1,dtype='int')]
                    #edge_inds = edge_point_inds[inds]
                    # Find ridges that divide the point at the centre of the current region
                    #region_simplex_inds_parallel = [x for ind in inds for k,x in enumerate(vor.ridge_points[:,0]) if x==ind]
                    region_simplex_inds_parallel = [point_simplex_inds[i] for i in inds]  # [np.where((vor.ridge_points[:,0]==ind) | (vor.ridge_points[:,1]==ind))[0] for ind in inds]
                    region_simplexes_parallel = [simplexes[i] for i in region_simplex_inds_parallel]
                    edge_point_index_parallel = edgepointInds[inds]
                    edge_region_parallel = [edge_regions[j] for j in inds]
                    vertices_parallel,coordinates_parallel = [],[]
                    
                    for j,ind in enumerate(inds):
                        #edge = graph.get_edge(edge_point_index_parallel[j])
                        edgeInd = edge_point_index_parallel[j]
                        vertices_parallel.append(vor.vertices[edgeInd])
                        npts = nedgepoints[edgeInd]
                        x0 = np.sum(nedgepoints[:edgeInd])
                        coordinates_parallel.append(edgepoints[x0:x0+npts])
                    
                    args[i] = [edge_region_parallel,vertices_parallel,coordinates_parallel,edge_point_index_parallel,inds,region_simplexes_parallel,region_simplex_inds_parallel,i]
                    ptr += n_reg_per_proc
                    i += 1
                
                breakpoint()
                test = find_voronoi_graph_intersections_multi(*args[0])
                breakpoint()
                res = pool.starmap(find_voronoi_graph_intersections_multi, args) # [proc,int/simp,data]
                intersections = [x for sublist in res for x in sublist[0]]
                if True:
                    breakpoint()
                    intersecting_simplex = [x for sublist in res for x in sublist[1]]
                else:
                    #intersecting_simplex = np.zeros(len(intersections),dtype='int') - 1
                    print('Identifying simplexes associated with intersections...')
                    breakpoint()
                    
                    sind = [np.where( ((simplexes[:,0]==ints['vertices'][0]) & (simplexes[:,1]==ints['vertices'][1])) | ((simplexes[:,0]==ints['vertices'][1]) & (simplexes[:,1]==ints['vertices'][0]))) for ints in tqdm(intersections)]
                    sind = arr(sind,dtype='int').flatten()
                    simplex_intersects_graph = np.zeros(simplexes.shape[0],dtype='bool')
                    simplex_intersects_graph[sind] = True
                    
                    #for j in trange(len(intersections)):
                    #    sind = np.where( ((simplexes[:,0]==intersections[j]['vertices'][0]) & (simplexes[:,1]==intersections[j]['vertices'][1])) | ((simplexes[:,0]==intersections[j]['vertices'][1]) & (simplexes[:,1]==intersections[j]['vertices'][0])))
                    #    if len(sind[0])>1:
                    #        breakpoint()
                    #    intersecting_simplex[j] = sind[0][0]
                    #    simplex_intersects_graph[sind[0]] = True
        else:
            # Indices of gridpoints corresponding to edges (rather than generated points)
            vor_points_derived_from_edges = edge_point_inds
            
             # Find ridges that divide the point at the centre of the current region
            #region_simplex_inds_parallel = [x for ind in inds for k,x in enumerate(vor.ridge_points[:,0]) if x==ind]
            region_simplex_inds = [point_simplex_inds[i] for i in vor_points_derived_from_edges]
            region_simplexes = [simplexes[i] for i in region_simplex_inds]
            # Filter edge points to only those present in the voronoi points
            # graph_edgepoint_edgeInds: indexes of the edge that each graph edgepoint belongs to 
            graph_edgepoint_edgeInds = graph.edgepoint_edge_indices()
            # vor_points_derived_from_edges: indexes of Voronoi points corresponing to edge points
            graph_edgepoint_inds_filt = graph_edgepoint_edgeInds[point_edgepoint_origin[point_type==1]] # [i for i,epi in enumerate(graph_edgepoint_edgeInds) if epi in vor_points_derived_from_edges]
            vertices = [vor.vertices[e] for e in edge_regions]
            coordinates = []
            # Loop through each Voronoi point associated with an edge
            for j,ind in enumerate(point_edgepoint_origin[point_type==1]):            
                # Get the global edge point index
                edgeInd = graph_edgepoint_edgeInds[ind]
                npts = nedgepoints[edgeInd]
                x0 = np.sum(nedgepoints[:edgeInd])
                crds = edgepoints[x0:x0+npts]
                coordinates.append(edgepoints[x0:x0+npts])

            """
            edge_region: vertex indices for the Voronoi edge (edge_regions)
            vertices: coordinates for each vertex in the Voronoi edge (vertices)
            coordinates: edgepoint coordiantes for the current graph edge (coordinates)
            edge_point_index: which graph edge (index) each the edgepoints come from  (graph_edgepoint_inds_filt)
            region_index: which region (index) the Voronoi edges come from (edge_point_inds)
            region_simplexes: Simplexes corresponding to each Voronoi edge
            region_simplex_inds: Indices of simplexes corresponding to each Voronoi edge
            """
            #breakpoint()
            res = find_voronoi_graph_intersections_multi(edge_regions,vertices,coordinates,graph_edgepoint_inds_filt, edge_point_inds,region_simplexes,region_simplex_inds,0)
            
            #for v in vertices: plt.plot(v[:,0],v[:,1],c='b')
            #for v in coordinates: plt.plot(v[:,0],v[:,1],c='r')
            #plt.show()
            
            intersections = res[0]
            intersecting_simplex = arr(res[1]).flatten()
            simplex_intersects_graph = np.zeros(simplexes.shape[0],dtype='bool')
            simplex_intersects_graph[intersecting_simplex] = True

        # Record which simplexes intersect graph edges
        #intersecting_simplex = arr(intersecting_simplex)
        #simplex_intersects_graph = np.zeros(simplexes.shape[0],dtype='bool')
        #simplex_intersects_graph[intersecting_simplex] = True

        # Filter simplices - only keep vertices inside volume
        if False:
            sind = np.where((simplexes[:,0]>=0) & (simplexes[:,1]>=0) & (remove_simps==False))
            simplexes = simplexes[sind]
            simplex_intersects_graph = simplex_intersects_graph[sind]
            
            # Remove simplexes that intersect edges
            simplexes = simplexes[~simplex_intersects_graph]

    # 2D to 3D
    vor_verts = np.hstack([vor_verts,np.zeros([vor_verts.shape[0],1])]).astype('float32')
    
    # Remove vertices outside volume
    #if radial_extent is None:
    #    radial_extent = 35000./2.
    #if radial_centre is None:
    #    radial_centre = arr([0.,0.,0.])
    keep = inside_vol(vor_verts,extent,centre=radial_centre,radius=radial_extent)
    #vor_verts, simplexes, lookup = update_array_index(vor_verts,simplexes,keep)

    # Remove connections that span the midline
    if avoid_midline and eye is not None:
        test = test_midline_collision(eye,vor_verts,simplexes,extent)
        simplexes = simplexes[~test]
        
    # Remove connections and vertices from inside the macula and optic nerve
    if avoid_macula and eye is not None:
        # Macula
        distances = np.linalg.norm(vor_verts-eye.fovea_centre,axis=1)
        fovea_radius = eye.fovea_radius #500. # um
        sind = np.where(distances<fovea_radius)
        #keep = np.ones(vor_verts.shape[0],dtype='bool')
        keep[sind] = False
        #vor_verts, simplexes, lookup = update_array_index(vor_verts,simplexes,keep)
        
    # Optic disc
    if avoid_optic_disc and eye is not None:
        distances = np.linalg.norm(vor_verts-eye.optic_disc_centre,axis=1)
        od_radius = eye.optic_disc_radius # um
        sind = np.where(distances<od_radius)
        #keep = np.ones(vor_verts.shape[0],dtype='bool')
        keep[sind] = False
        #vor_verts, simplexes, lookup = update_array_index(vor_verts,simplexes,keep)

    sind = np.where((simplexes[:,0]>=0) & (simplexes[:,1]>=0) )
    simplexes = simplexes[sind]
    vor_verts, simplexes, lookup = update_array_index(vor_verts,simplexes,keep)

    # Update intersection vertex indices
    valid = np.ones(len(intersections),dtype='bool')
    print('Updating vertex indices...')
    for i,inter in enumerate(tqdm(intersections)):
        inter['vertices'][0] = lookup[inter['vertices'][0]]
        inter['vertices'][1] = lookup[inter['vertices'][1]]
        if np.any(arr(inter['vertices'])==-1):
            valid[i] = False
    intersections = [x for i,x in enumerate(intersections) if valid[i]==True]
    nintersections = len(intersections)
    
    # Update endpoint connections
    valid = np.ones(len(endpoint_conn),dtype='bool')
    print('Updating endpoint connections...')
    for i,endp in enumerate(tqdm(endpoint_conn)):
        endp['vertex'] = lookup[endp['vertex']]
        if endp['vertex']==-1:
            valid[i] = False
    endpoint_conn = [x for i,x in enumerate(endpoint_conn) if valid[i]==True]
    nendpoint_conn = len(endpoint_conn)
    
    # Find unique simplex vertices
    vert_inds = np.unique(simplexes)
    vert_count = np.bincount(simplexes.flatten())

    if False:
        plot_voronoi(vor_verts,simplexes,graph=graph,meshes=None)
    
    # Storage object          
    gvars = GVars(graph)

    # Add pre-edited voroni vertices as nodes
    gvars.gnode_0 = 0 # start of graph nodes
    # Record start of voronoi vertex nodes (i.e. end of graph node list)
    gvars.vnode_0 = gvars.node_ptr #nodecoords.shape[0] 
    gvars.append_nodes(vor_verts,update_pointer=True)
    # Record start of additional nodes list (to be added below)
    gvars.enode_0 = gvars.node_ptr #gvars.nodecoords.shape[0]
    # Add edited simplexes from vornoi into graph, offset to location of vertices in node list
    gvars.gedge_0 = 0 # start of graph edges
    gvars.vedge_0 = gvars.edge_ptr
    
    scnames = [x['name'] for x in graph.get_scalars()]
    sctype = [x['name'] for x in graph.get_scalars()]
    radname = graph.get_radius_field_name()
    radInd = scnames.index(radname)
    vtInd = scnames.index('VesselType')

    print('Adding non-intersecting simplexes to graph...')
    new_scalar_vals = np.zeros(len(gvars.scalars))
    new_scalar_vals[radInd] = capillary_radius # radius
    new_scalar_vals[vtInd] = 2 # VesselType
    for i,simp in enumerate(tqdm(simplexes)):
        cur_econn = simp+gvars.vnode_0
        gvars.add_edge(cur_econn[0],cur_econn[1],new_scalar_vals)
    # Record end of simplex edges
    gvars.cedge_0 = gvars.edge_ptr

    # Add in endpoint connections
    print(f'Connecting {len(endpoint_conn)} endpoints...')
    for epc in tqdm(endpoint_conn):
        edge = graph.get_edge(epc['edge_index'])
        scvals = [x[0] for x in edge.scalars]
        #scvals = [edge.scalars[0][0],edge.scalars[1][0],0.] # Radius, VesselType, mlp
        # Add connection - offset vornoi vertex by vornoi start index
        c1, c2 = epc['node'],epc['vertex']+gvars.vnode_0 
        if c1 not in inlets and c1 not in outlets and c2 not in inlets and c2 not in outlets:
            gvars.add_edge(c1, c2, scvals)

    ### Create new edges from simplex intersections with graph ### 
    edges_intersected = arr([x['edge'] for x in intersections])
    unq_edges_intersected,intersection_count = np.unique(edges_intersected,return_counts=True)

    remove_edge_inds = []
    # Loop through all intersected edges
    if find_intersections:
        print('Creating intersected edges...') # slow for large networks... (~40 mins for 2M edges)
    
        for eind in tqdm(unq_edges_intersected):
            cur_intersections = [x for x in intersections if x['edge']==eind]
            cur_pts = arr([x['intersection'] for x in cur_intersections])
            
            edge = graph.get_edge(eind)
            edge_radii = edge.scalars[edge.scalarNames==gvars.radname]
            edge_rad_val = np.mean(edge_radii)
            
            if edge_rad_val<=max_connection_vessel_radius:
                start_node = edge.start_node_coords
                end_node = edge.end_node_coords
                scvals = [x[0] for x in edge.scalars]
                scvals[edge.scalarNames==gvars.radname] = edge_rad_val
                
                scvalsCap = scvals.copy()
                scvalsCap[edge.scalarNames==gvars.radname] = capillary_radius
                cap_conn = False
                if cap_conn: # Connect as capillary
                    scvalsCap[1] = 2 # VesselTsype=capillary
                else: # connect as current type
                    scvalsCap[1] = edge.scalars[1][0]
                
                # Ordering of new points
                dists = np.linalg.norm(cur_pts-start_node,axis=1)
                srt = np.argsort(dists)
                cur_intersections = [cur_intersections[i] for i in srt]
                cur_pts = cur_pts[srt]
                
                count = 0
                for i,pt in enumerate(cur_pts):
                
                    # Check for degeneracy. If not, carry on...
                    sind = np.where((gvars.nodecoords[:,0]==pt[0]) & (gvars.nodecoords[:,1]==pt[1]) & (gvars.nodecoords[:,2]==pt[2]))
                    if len(sind[0])==0:
                    #if True:
                    
                        # Create a node
                        gvars.add_node(pt)
                  
                        # Replicate edge
                        if count==0:
                            # Connect new node to start node
                            gvars.add_edge(edge.start_node_index, gvars.node_ptr-1, scvals)
                        else:
                            # Connect new node to previous node
                            gvars.add_edge(gvars.node_ptr-2, gvars.node_ptr-1, scvals)

                        # Connect to Voronoi
                        if np.random.uniform()>0.5: # Only add in a fraction of edges
                            if np.random.uniform()>0.5: # Decide left or right connection
                                gvars.add_edge(gvars.node_ptr-1, cur_intersections[i]['vertices'][0]+gvars.vnode_0, scvalsCap)
                            else:
                                gvars.add_edge(gvars.node_ptr-1, cur_intersections[i]['vertices'][1]+gvars.vnode_0, scvalsCap)
                                    
                        count += 1

                # Add final connection point at end
                gvars.add_edge(gvars.node_ptr-1, edge.end_node_index, scvals)
                
                remove_edge_inds.append(eind)
            
            if False:
                plt.scatter(cur_pts[:,0],cur_pts[:,1])
                plt.scatter(start_node[0],start_node[1])
                plt.scatter(end_node[0],end_node[1])
                plt.show()

        # Remove old edges
        remove_edge_inds = arr(remove_edge_inds)
        
        graph = gvars.set_in_graph()  
        # Remove any remaining edges
        if len(remove_edge_inds)>0:
            gvars.remove_edges(remove_edge_inds)
    else:
        graph = gvars.set_in_graph()  
    
    if radial_extent is not None and False: 
        gc = graph.get_node_count()
        stmp = np.where(gc==1)
        nodecoords = graph.get_data('VertexCoordinates')
        dist = np.linalg.norm(nodecoords-radial_centre,axis=1)
        keepnodes = np.where((dist>radial_extent) & (gc==1))
        graph = remove_all_endpoints(graph,keep=np.concatenate([inlets,outlets,keepnodes[0]]))
    
        graph = connect_vessel_endpoints_directly(graph)    
    else: 
        #graph = remove_all_endpoints(graph,keep=np.concatenate([inlets,outlets]))
        pass

    if False:
        cyls = graph.plot_graph(radius_scale=1.,scalar_color_name='radius',cmap='gray') #,edge_filter=edge_filter)

    return graph
    
def create_capillary_voronoi_simple(graph,eye=None,regular_mesh=False,voronoi_vessel_avoidance=True,avoid_midline=True,avoid_macula=True,avoid_optic_disc=False,spacing=150.,simplex_graph_collision_resolution=True,remove_voronoi_nodes_inside_vessels=True,write=True,ofile=None,capillary_radius=3.,max_connection_vessel_radius=20.,radial_extent=None,radial_centre=None):

    """
    As above, but creates a Vornonoi networks and simply connects graph endpoints to it
    """

    print('Calculating voronoi (simple)')

    # Identify arterial inlets (2) and venous outlets (2)
    inlet1,outlet1 = identify_inlet_outlet(graph)
    inlet2,outlet2 = identify_inlet_outlet(graph,ignore=[inlet1,outlet1])
    if inlet2 is not None:
        inlets = [inlet1,inlet2]
    else:
        inlets = [inlet1]
    if outlet2 is not None:
        outlets = [outlet1,outlet2]
    else:
        outlets = [outlet1]

    # Create lightweight object for editing graph            
    gvars = GVars(graph)
    extent = np.asarray(graph.edge_spatial_extent())

    # Create Voronoi points
    grid_points = voronoi_points(graph=graph,regular_mesh=False,extent=extent,spacing=spacing,eye=eye,radial_extent=radial_extent,radial_centre=radial_centre)

    points = grid_points
    npoints = points.shape[0]

    vor = Voronoi(points[:,0:2],qhull_options='Qbb Qc Qx')
    
    vor_verts = vor.vertices
    simplexes = arr(vor.ridge_vertices)
    
    # Remove boundary connections
    sind = np.where( (simplexes[:,0]>=0) & (simplexes[:,1]>=0) )
    simplexes = simplexes[sind]
    
    regions = [vor.regions[vor.point_region[i]] for i in range(npoints)] 
    
    # Identify node branch number
    gc = graph.get_node_count()
    # Create endpoint connections
    send = np.where(gc==1)   
    endpoint_conn = []
    print('Calculating endpoint connections...')
    # Loop through each endpoint
    remove_simps = np.zeros(simplexes.shape[0],dtype='bool')
    for si in tqdm(send[0]):
        edgeInd = graph.get_edges_containing_node(si)
        edge = graph.get_edge(edgeInd[0])
        if True: #if edge.i0 in edgepoint_lookup and edge.i1 in edgepoint_lookup:
            if edge.start_node_index==si:
                pt1 = edge.start_node_coords
                pt0 = edge.coordinates[1]
                dr = pt1 - pt0
            elif edge.end_node_index==si:
                pt1 = edge.end_node_coords
                pt0 = edge.coordinates[-2]
                dr = pt1 - pt0
            else:
                breakpoint()
                
            # Find closest vertex
            vor_reg_verts = vor.vertices #[edge_region]
            dists = np.linalg.norm(vor_reg_verts-pt1[:2],axis=1)
            
            vor_reg_verts_inds = np.where(dists<spacing*3)
            if len(vor_reg_verts_inds[0])==0:
                breakpoint()
            vor_reg_verts = vor_reg_verts[vor_reg_verts_inds]
            dists = dists[vor_reg_verts_inds]
            
            #angles = geometry.vector_angle(dr[:2],vor_reg_verts-pt1[:2])
            if np.any(dists==0) or np.linalg.norm(dr)==0.:
                breakpoint()
            arg = np.dot(np.reshape(dr[:2],[1,2]),(vor_reg_verts-pt1[:2]).transpose()).squeeze() #/ (dists*np.linalg.norm(dr))
            conn_ind = np.argmax(arg/dists)
            #arg = np.clip(arg,-1,1)
            #angles = np.arccos(arg)
            #conn_ind = np.argmin(arg)
            
            # To avid hyper-connected nodes, if more than 2 connections identified, remove one
            stmp = np.where((simplexes[:,0]==vor_reg_verts_inds[0][conn_ind]) | (simplexes[:,1]==vor_reg_verts_inds[0][conn_ind]))
            #breakpoint()
            if len(stmp[0])>2:
                simpi = simplexes[stmp]
                remove_simps[np.random.choice(stmp[0])] = True
            endpoint_conn.append({'vertex':vor_reg_verts_inds[0][conn_ind],'node':si,'edge_index':edgeInd[0]})
     
    intersections = []
    intersecting_simplex = []

    # 2D to 3D
    vor_verts = np.hstack([vor_verts,np.zeros([vor_verts.shape[0],1])]).astype('float32')
    
    # Remove vertices outside volume
    keep = inside_vol(vor_verts,extent,centre=radial_centre,radius=radial_extent)
    #vor_verts, simplexes, lookup = update_array_index(vor_verts,simplexes,keep)

    # Remove connections that span the midline
    if avoid_midline and eye is not None:
        test = test_midline_collision(eye,vor_verts,simplexes,extent)
        simplexes = simplexes[~test]
        
    # Remove connections and vertices from inside the macula and optic nerve
    if avoid_macula and eye is not None:
        # Macula
        distances = np.linalg.norm(vor_verts-eye.fovea_centre,axis=1)
        fovea_radius = eye.fovea_radius #500. # um
        sind = np.where(distances<fovea_radius)
        #keep = np.ones(vor_verts.shape[0],dtype='bool')
        keep[sind] = False
        #vor_verts, simplexes, lookup = update_array_index(vor_verts,simplexes,keep)
        
    # Optic disc
    if avoid_optic_disc and eye is not None:
        distances = np.linalg.norm(vor_verts-eye.optic_disc_centre,axis=1)
        od_radius = eye.optic_disc_radius # um
        sind = np.where(distances<od_radius)
        #keep = np.ones(vor_verts.shape[0],dtype='bool')
        keep[sind] = False
        #vor_verts, simplexes, lookup = update_array_index(vor_verts,simplexes,keep)
    
    vor_verts, simplexes, lookup = update_array_index(vor_verts,simplexes,keep)
    
    # Update endpoint connections
    valid = np.ones(len(endpoint_conn),dtype='bool')
    print('Updating endpoint connections...')
    for i,endp in enumerate(tqdm(endpoint_conn)):
        endp['vertex'] = lookup[endp['vertex']]
        if endp['vertex']==-1:
            valid[i] = False
    endpoint_conn = [x for i,x in enumerate(endpoint_conn) if valid[i]==True]
    nendpoint_conn = len(endpoint_conn)
    
    # Find unique simplex vertices
    #vert_inds = np.unique(simplexes)
    #vert_count = np.bincount(simplexes.flatten())

    if False:
        plot_voronoi(vor_verts,simplexes,graph=graph,meshes=None)
    
    # Storage object          
    gvars = GVars(graph)
    # Add pre-edited voroni vertices as nodes
    gvars.gnode_0 = 0 # start of graph nodes
    # Record start of voronoi vertex nodes (i.e. end of graph node list)
    gvars.vnode_0 = gvars.node_ptr #nodecoords.shape[0] 
    gvars.append_nodes(vor_verts,update_pointer=True)
    # Record start of additional nodes list (to be added below)
    gvars.enode_0 = gvars.node_ptr #gvars.nodecoords.shape[0]
    # Add edited simplexes from vornoi into graph, offset to location of vertices in node list
    gvars.gedge_0 = 0 # start of graph edges
    gvars.vedge_0 = gvars.edge_ptr

    scnames = [x['name'] for x in graph.get_scalars()]
    sctype = [x['name'] for x in graph.get_scalars()]
    radname = graph.get_radius_field_name()
    radInd = scnames.index(radname)
    vtInd = scnames.index('VesselType')

    print('Adding simplexes to graph...')
    new_scalar_vals = np.zeros(len(gvars.scalars))
    new_scalar_vals[radInd] = capillary_radius # radius
    new_scalar_vals[vtInd] = 2 # VesselType
    for i,simp in enumerate(tqdm(simplexes)):
        cur_econn = simp+gvars.vnode_0
        gvars.add_edge(cur_econn[0],cur_econn[1],new_scalar_vals)
    # Record end of simplex edges
    gvars.cedge_0 = gvars.edge_ptr

    # Add in endpoint connections
    if True:
        print('Connecting endpoints...')
        for epc in tqdm(endpoint_conn):
            edge = graph.get_edge(epc['edge_index'])
            scvals = [x[0] for x in edge.scalars]
            #scvals = [edge.scalars[0][0],edge.scalars[1][0],0.] # Radius, VesselType, mlp
            # Add connection - offset vornoi vertex by vornoi start index
            c1, c2 = epc['node'],epc['vertex']+gvars.vnode_0 
            if c1 not in inlets and c1 not in outlets and c2 not in inlets and c2 not in outlets:
                gvars.add_edge(c1, c2, scvals)
            
    graph = gvars.set_in_graph() 
    
    if radial_extent is not None and False: 
        gc = graph.get_node_count()
        stmp = np.where(gc==1)
        nodecoords = graph.get_data('VertexCoordinates')
        dist = np.linalg.norm(nodecoords-radial_centre,axis=1)
        keepnodes = np.where((dist>radial_extent) & (gc==1))
        graph = remove_all_endpoints(graph,keep=np.concatenate([inlets,outlets,keepnodes[0]]))
    
        graph = connect_vessel_endpoints_directly(graph)    
    else: 
        graph = remove_all_endpoints(graph,keep=np.concatenate([inlets,outlets]))

    
    breakpoint()
    
    if False:
        cyls = graph.plot_graph(radius_scale=1.,scalar_color_name='radius',cmap='gray') #,edge_filter=edge_filter)

    return graph    

def restore_voronoi(vfile):
    data = np.load(vfile)
    vor_verts = data['arr_0']
    simplexes = data['arr_1']
    return vor_verts, simplexes
    
def get_degenerate_node_pairs(graph):

    nodecoords_new,edgeconn_new,edgepoints_new,nedgepoints_new,radius_new,category_new,mlp_new = get_graph_fields(graph)

    degen_pairs = []
    for i,node in enumerate(tqdm(nodecoords_new)):
        mtch = np.where(np.all(node==nodecoords_new,axis=1))
        mtch = arr([x for x in mtch[0] if x!=i])
        if len(mtch)>0:
            for m in mtch:
                degen_pairs.append([i,m])

    degen_pairs = arr(degen_pairs)
    
    return degen_pairs
    
def displace_degen_nodes(graph,distance=10.,dispall=True):

    nodecoords,edgeconn,edgepoints,nedgepoints,radius,category,mlp = get_graph_fields(graph)

    # Just slightly displace all nodes by a random amount
    if dispall:
        nodecoords[:,0:2] = nodecoords[:,0:2] + np.random.uniform(-distance/2.,distance/2.,size=[nodecoords.shape[0],2])
        x0 = np.cumsum(nedgepoints,dtype='int') - nedgepoints[0]
        x1 = x0 + nedgepoints - 1
        edgepoints[x0.astype('int')] = nodecoords[edgeconn[:,0]]
        edgepoints[x1.astype('int')] = nodecoords[edgeconn[:,1]]
    
        #visited = np.zeros(graph.nnode,dtype='bool')
        #for i,ec in enumerate(tqdm(edgeconn)):
        #    x0 = np.sum(nedgepoints[:i])
        #    x1 = x0 + nedgepoints[i]
        #    
        #    if not visited[ec[0]]:
        #        nodecoords[ec[0],0:2] = nodecoords[ec[0],0:2] + np.random.uniform(-distance/2.,distance/2.,2)
        #        visited[ec[0]] = True
        #    if not visited[ec[1]]:
        #        nodecoords[ec[1],0:2] = nodecoords[ec[1],0:2] + np.random.uniform(-distance/2.,distance/2.,2)
        #        visited[ec[1]] = True
        #
        #    edgepoints[x0] = nodecoords[ec[0]]
        #    edgepoints[x1-1] = nodecoords[ec[1]]

        graph = set_and_write_graph(graph, [nodecoords,edgeconn,edgepoints,nedgepoints,radius,category,mlp],None)
    # Otherwise look for degenerate pairs and displace all but one (buggy...)
    else:
        degen_pairs = get_degenerate_node_pairs(graph)
        
        visited = np.zeros(nodecoords.shape[0],dtype='int')
        for pair in tqdm(degen_pairs):
            if visited[pair[1]]==0:
                nodecoords[pair[1],0] += np.random.uniform(-distance/2.,distance/2.)
                nodecoords[pair[1],1] += np.random.uniform(-distance/2.,distance/2.)
                visited[pair[0]] = 1 # visited and not moved
                visited[pair[1]] = 2 # visited and moved
                
                # Move edgepoints too
                sind = np.where((edgeconn[:,0]==pair[1]) | (edgeconn[:,1]==pair[1]))
                for iedge in sind[0]:
                    x0 = np.sum(nedgepoints[:iedge])
                    x1 = x0 + nedgepoints[iedge]
                    
                    if edgeconn[iedge,0]==pair[1]:
                        edgepoints_new[x0,0] += distance
                        edgepoints_new[x0,1] += distance
                    elif edgeconn[iedge,1]==pair[1]:
                        edgepoints_new[x1-1,0] += distance
                        edgepoints_new[x1-1,1] += distance

        graph = set_and_write_graph(graph, [nodecoords,edgeconn,edgepoints_new,nedgepoints,radius,category,mlp],None)
    return graph
    
def connect_vessel_endpoints_directly(graph,rad=0.1,furthest_node_distance=1000.,max_connectivity=3,mxDist=1e12,restore_edge_lookup=False,eye=None,ninlets=2):

    print('Directly connecting arterial terminal nodes to venous terminal nodes')
    
    """
     inlets: indices of nodes acting as inlets
     outlets: indices of nodes acting as outlets
     a_inds: node indices for veinous endpoints
     v_inds: node indices for veinous endpoints
     rad: radius assigned to new connections (um)
     furthest_node_distance: furthest distance between nodes for a connection to be made
     mxDist: largest distance to temporarily assign to points that will never be matched
    """
    
    nodecoords_new,edgeconn_new,edgepoints_new,nedgepoints_new,radius_new,category_new,mlp_new = get_graph_fields(graph)

    extent = np.asarray(graph.edge_spatial_extent())
    #ngraph_coords = nodecoords.shape[0]

    inds = np.linspace(0,edgeconn_new.shape[0]-1,edgeconn_new.shape[0],dtype='int')
    edge_inds = np.repeat(inds,nedgepoints_new)
    first_edgepoint_inds = np.concatenate([[0],np.cumsum(nedgepoints_new)[:-1]])
    
    # Find where each node appears in the edge connectivity array, and store for future use
    tmpfile = '~/tmp.p'
    edge_node_lookup = create_edge_node_lookup(nodecoords_new,edgeconn_new,tmpfile=tmpfile,restore=restore_edge_lookup)

    # Assign a radius to nodes using the largest radius of each connected edge
    edge_radius = radius_new[first_edgepoint_inds]
    node_radius = arr([np.max(edge_radius[edge_node_lookup[i]],initial=-1.) for i in range(nodecoords_new.shape[0])])

    # Assign a category to each node using the minimum category of each connected edge (thereby favouring arteries/veins (=0,1) over capillaries (=2))
    edge_category = category_new[first_edgepoint_inds]
    node_category = arr([np.min(edge_category[edge_node_lookup[i]],initial=2) for i in range(nodecoords_new.shape[0])])
    
    #if vor_verts is None:
    #keep_node = np.zeros_like(node_category,'bool')
    #keep_node[node_category==2] = True
    #vor_verts,simplexes,lookup = update_array_index(nodecoords_new,edgeconn_new,keep_node)

    # Define value to assign as the largest possible distance
    mxDist = np.max(extent[:,1]-extent[:,0]) * 2
    
    # Identify arterial inlets (2) and venous outlets (2)
    inlet1,outlet1 = identify_inlet_outlet(graph)
    if ninlets==2:
        inlet2,outlet2 = identify_inlet_outlet(graph,ignore=[inlet1,outlet1])
        inlets = [inlet1,inlet2]
        outlets = [outlet1,outlet2]
    else:
        inlets = [inlet1]
        outlets = [outlet1]
    
    # Find graph end nodes (1 connection only)
    #term_inds = np.where((g_count==1) & ((node_category==0) | (node_category==1)))
    #terminal_node = np.zeros(nodecoords_new.shape[0],dtype='bool')
    #terminal_node[term_inds] = True
        
    count = 0
    unconnected_endpoints = []
    for j in range(2):
            
        # Calculate node connectivity
        g_count = get_node_count(graph,edge_node_lookup=edge_node_lookup)

        v_inds = np.where((node_category==1) & (g_count==1))[0]
        v_inds = v_inds[(~np.in1d(v_inds,inlets)) & (~np.in1d(v_inds,outlets))]
        a_inds = np.where((node_category==0) & (g_count==1))[0]
        a_inds = a_inds[(~np.in1d(a_inds,inlets)) & (~np.in1d(a_inds,outlets))]
        
        nodecoords_new,edgeconn_new,edgepoints_new,nedgepoints_new,radius_new,category_new,mlp_new = get_graph_fields(graph)
        
        # How many connections associated with each node
        graph_params = [nodecoords_new,edgeconn_new,edgepoints_new,nedgepoints_new,radius_new,category_new,mlp_new]
        g_count = get_node_count(None,graph_params=graph_params)
        # How many connections associated with arteries
        a_count = g_count[a_inds]
        # How many connections associated with veins
        v_count = g_count[v_inds]
    
        if j==0:
            source_inds = a_inds
            target_inds = v_inds
            target_count = v_count
            print('Connecting arterial endpoints to venous endpoints')
        elif j==1:
            source_inds = v_inds
            target_inds = a_inds
            target_count = a_count
            print('Connecting venous endpoints to arterial endpoints')

        for i,s_node_ind in enumerate(tqdm(source_inds)):    
  
            # Calculate the distance from each venous vertex to the current arterial endpoint
            distances = np.linalg.norm(nodecoords_new[target_inds]-nodecoords_new[s_node_ind],axis=1)
            # Any edgepoints that aren't inline or are too far away (or too close) assign a large distance to (used to be a NaN, but proved problematic...)
            distances[(target_count>=max_connectivity) | (distances>furthest_node_distance)] = mxDist
            #distances[outlets] = mxDist
            # Find the closest edgepoint to the current vertex
            dmin = np.nanargmin(distances)
            
            if distances[dmin]<mxDist:               
                # Connect new node to voronoi vert
                t_node_ind = dmin
                #new_edge = arr([[t_node_ind+ngraph_coords,g_node_ind]])
                new_edge = arr([[target_inds[t_node_ind],s_node_ind]])
                new_points = arr([nodecoords_new[target_inds][t_node_ind],nodecoords_new[s_node_ind]])
                nnew_points = arr([2])
                #new_radii = arr([node_radius[target_inds][t_node_ind],node_radius[s_node_ind]])
                new_radii = arr([rad,rad])
                new_category = arr([2,2])
                new_mlp = arr([0,0])

                # Add the new edgepoints to the storage arrays
                edgeconn_new = np.concatenate([edgeconn_new,new_edge]) 
                edgepoints_new = np.concatenate([edgepoints_new,new_points])
                radius_new = np.concatenate([radius_new,new_radii])
                nedgepoints_new = np.concatenate([nedgepoints_new,nnew_points])
                category_new = np.concatenate([category_new,new_category])
                mlp_new = np.concatenate([mlp_new,new_mlp])

                count += 1
                target_count[dmin] += 1
            else:
                #print('Could not connect arterial endpoint node')
                unconnected_endpoints.append(s_node_ind)
           
        # Update graph         
        graph = set_and_write_graph(graph, [nodecoords_new,edgeconn_new,edgepoints_new,nedgepoints_new,radius_new,category_new,mlp_new],None)
            
    #if len(unconnected_endpoints)>0:
    #    print(f'{len(unconnected_endpoints)} unconnected graph endpoints remaining!')

    graph = set_and_write_graph(graph, [nodecoords_new,edgeconn_new,edgepoints_new,nedgepoints_new,radius_new,category_new,mlp_new],None)
    graph = remove_all_endpoints(graph,keep=np.concatenate([inlets,outlets]))

    # Remove midline crossing points
    if False and eye is not None:
        nodecoords, edgeconn, edgepoints, nedgepoints, radius, category, mlp = get_graph_fields(graph)
        extent = arr(graph.edge_spatial_extent())
        test = test_midline_collision(eye,nodecoords,edgeconn,extent)
        gvars = GVars(graph)
        gvars.remove_edges(np.where(test)[0])
        graph = gvars.set_in_graph()  
        breakpoint()    
        graph = remove_all_endpoints(graph,keep=np.concatenate([inlets,outlets]))
        
    return graph
    
def voronoi_capillary_bed(path,g_file_init,graph=None,write=True,plot=True,displace_degen=True,geometry_file=None,eye=None,capillary_radius=3.):

    ### OPTIONS ###
    restore_combined = False # Restore saved combined file (bypasses most steps!)
    filter = False # Load in graph, filter by size then fit a voronoi network around it
    restore_filter = True
    create_voronoi = True # Generate Voronoi, otherwise restore from default file
    voronoi_vessel_avoidance = True # use vessel avoidance algorithms to edit Voronoi
    simplex_graph_collision_resolution = True # (only if create_voronoi=True)
    remove_voronoi_nodes_inside_vessels = True # (only if create_voronoi=True)
    #interpolate_vessels = True
    connect_vessels_to_capillaries = True
    connect_capillaries_to_vessel_segments = True
    restore_edge_lookup = False # Use pickled edge_lookup data or not
    
    filter_clip_radius = 0. # 0. um. Any vessel radii below this radius get clipped to the minimum value below 
                                 # (i.e. vessels with min_filter_radius<radius<filter_clip_radius -> radius=filter_clip_radius; then radius<min_filter_radius are deleted
    min_filter_radius = 8. # um. After clipping, vessels with radii below this value get deleted
    regular_mesh = False # regular or random Voronoi points
    spacing = 50. #150. #um Target spacing between Voronoi points (or regular mesh)
    interpolation_resolution = 100. #200. # um
    max_connection_vessel_radius = 20. #1e6 #um The largest vessel radius that a capillary can connect to (TODO: make probabilistic)
    furthest_node_distance = spacing*10 #1000. #um The greatest distance between nodes that can be connected
    ###############
    
    # Files
    # Filtered graph file
    #g_filt_file = join(path,g_file_init.replace('.am','_voronoi_filtered.am'))
    # Final output file
    g_combined_file = join(path,g_file_init.replace('.am','_voronoicap.am'))
    
    # Voronoi output (numpy pz file)
    vfile = join(path,'voronoi.npz')

    # LOAD EYE GEOMETRY FILE   
    if eye is None:
        # Eye geometry file (L-system output)
        if geometry_file is None:
            geometry_file = join(path,'retina_geometry.p')
        eye = Eye()
        eye.load(geometry_file)
    radial_extent = (eye.domain[0,1] - eye.domain[0,0])/2.
    radial_centre = np.mean(eye.domain,axis=1)
    
    if restore_combined:
        print(f'Importing combined graph: {g_combined_file}')
        connected_graph = spatialgraph.SpatialGraph(initialise=True)
        connected_graph.read(g_combined_file)

    # FILTER GRAPH BY VESSEL RADIUS (or load pre-filtered)
    if graph is None:
        print(f'Importing graph: {g_file_init}')
        graph = spatialgraph.SpatialGraph(initialise=True)
        graph.read(join(path,g_file_init))
    
    if filter:        
        print(f'Filtering graph by radius. Min: {min_filter_radius}, clip: {filter_clip_radius}')
        filter_graph_by_radius(graph,min_filter_radius=min_filter_radius,filter_clip_radius=filter_clip_radius)
        
        # After filtering, remove excees inline nodes
        ed = spatialgraph.Editor()
        graph = ed.remove_intermediate_nodes(graph)
        
        # INTERPOLATE VESSELS before connecting voronoi verts     
        print(f'Interpolating graph ({interpolation_resolution} um)')
        ed.interpolate_edges(graph,interp_resolution=interpolation_resolution,filter=None,noise_sd=0.)

        # Write
        #graph.write(g_filt_file)

    # CREATE VORONOI 
    if True:
        connected_graph = create_capillary_voronoi(graph,eye=eye,regular_mesh=regular_mesh,voronoi_vessel_avoidance=voronoi_vessel_avoidance,avoid_midline=False,avoid_macula=True,avoid_optic_disc=False,spacing=spacing,simplex_graph_collision_resolution=simplex_graph_collision_resolution,remove_voronoi_nodes_inside_vessels=remove_voronoi_nodes_inside_vessels,write=True,ofile=vfile,max_connection_vessel_radius=max_connection_vessel_radius,radial_extent=radial_extent,radial_centre=radial_centre,capillary_radius=capillary_radius)
    
    print('Calculating lengths...')
    lengths = connected_graph.get_node_to_node_lengths()
    #breakpoint()
    stmp = np.where(lengths==0.)
    print(f'{len(stmp[0])} zero-length segments detected')
    count = 0
    displace_degen = False
    if np.any(lengths<=0) and displace_degen:
        while True:
            print('Displacing degenerate nodes...')
            connected_graph = displace_degen_nodes(connected_graph,distance=0.5,dispall=True)
            break
            
            # Check node-node lengths
            if False:
                lengths = connected_graph.get_node_to_node_lengths()
                if np.any(lengths<=0):
                    print(f'Zero-length segments still remaining: {len(np.where(lengths<=0.)[0])}')
                    count += 1
                else:
                    break
                    
                if count>5:
                    breakpoint()
    
    if write:
        print(f'Writing graph: {g_combined_file}')
        connected_graph.write(g_combined_file)
    if plot:
        connected_graph.plot_graph()
        
    return connected_graph
    
def direct_arterial_venous_connection(path,gfile,ofile,eye=None,graph=None,displace_degen=True,write=True,geometry_file=None,ninlets=2):
        
    if eye is None:    
        if geometry_file is None:
            geometry_file = join(path,'retina_geometry.p')
        eye = Eye()
        eye.load(geometry_file)    

    if graph is None:
        graph = spatialgraph.SpatialGraph(initialise=True)
        graph.read(join(path,gfile))

    max_connection_vessel_radius = 1e6
    furthest_node_distance = 1500.

    print('Connecting endpoints...')
    graph = connect_vessel_endpoints_directly(graph,eye=eye,ninlets=ninlets)

    lengths = graph.get_node_to_node_lengths()
    count = 0
    if np.any(lengths<=0) and displace_degen:
        while True:
            print('Displacing degenerate nodes...')
            graph = displace_degen_nodes(graph,distance=1.)
            
            # Check node-node lengths
            lengths = graph.get_node_to_node_lengths()
            if np.any(lengths<=0):
                print(f'Zero-length segments still remaining: {len(np.where(lengths<=0.)[0])}')
                count += 1
            else:
                break
                
            if count>5:
                breakpoint()
        
    print(f'Shortest vessel length: {lengths.min()}')
    
    if write:
        print(f'Writing graph to {ofile}')
        graph.write(ofile)
    
    return graph
    
def remove_arteriole(graph,n=10,min_radius=0.,max_radius=20.,grab_file='',eye=None,min_displacement=0.,remove_endpoints=True,select_only=False,catId=0):

    nodecoords, edgeconn, edgepoints, nedgepoints, radius, category, mlp = get_graph_fields(graph)
    
    inlet1,outlet1 = identify_inlet_outlet(graph)
    #inlet2,outlet2 = identify_inlet_outlet(graph,ignore=[inlet1,outlet1])
    inlets = [inlet1]
    outlets = [outlet1]
    
    if eye is not None and min_displacement>=0.:
        dists = np.linalg.norm((edgepoints-eye.macula_centre).T,axis=0)
    else:
        dists = np.zeros(edgepoints.shape[0]) - 1.
    
    # Find arterioles 
    # catId=0 for arteries, 1 fro veins, 2 for capillaries
    sind = np.where((category==catId) & (radius>=min_radius) & (radius<=max_radius) & (dists>=min_displacement))
    if len(sind[0])>0:
        # Randomly select edgepoints within arterioles
        rind = np.random.choice(sind[0],size=n,replace=False)
        # Find all edgepoints associated with random edgepoints
        epi = graph.edgepoint_edge_indices()
        redge = np.unique(epi[rind])
    else:
        redge = []
    
    if grab_file!='':
        vis = graph.plot_graph(scalar_color_name='VesselType',show=False,block=False,edge_highlight=redge)
        for e in redge:
            edge = graph.get_edge(e)
            snc,enc = edge.start_node_coords,edge.end_node_coords
            centre = (snc+enc)/2.
            centre[2] = 200.
            length = np.linalg.norm(enc-snc)
            rad = np.clip(length*1.2,250.,None)
            vis.add_torus(centre=centre,torus_radius=rad,tube_radius=20.) 
        vis.screen_grab(grab_file)
    
    if select_only:
        return redge
    
    gvars = GVars(graph)
    gvars.remove_edges(redge)
    graph = gvars.set_in_graph()
    
    if remove_endpoints:
        graph = remove_all_endpoints(graph,keep=np.concatenate([inlets,outlets]))

    return graph
        
