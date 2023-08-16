import numpy as np
arr = np.asarray
from matplotlib import pyplot as plt
from vessel_sim import geometry
from scipy import interpolate
from tqdm import tqdm, trange

import os
join = os.path.join

from pymira import spatialgraph
from retinasim.utility import identify_inlet_outlet

zaxis = arr([0.,0.,1.])
xaxis = arr([1.,0.,0.])

"""
Set of functions for imposing sinosoidal path fluctuations (tortuosity) onto vessel networks
apply_to_graph takes a graph object, or filename, as an argument
"""

def linear_fill(values,ninterp=5):
    nintval = (values.shape[0]-1)*ninterp+1
    if len(values.shape)>1:
        values_int = np.zeros([nintval]+list(values.shape[1:]))
    else:
        values_int = np.zeros(nintval)
    flag = np.zeros(nintval,dtype='int') - 1
    for i in range(1,values.shape[0]):
        values_int[(i-1)*ninterp:((i-1)*ninterp)+ninterp+1] = np.linspace(values[i-1],values[i],ninterp+1)
        flag[(i-1)*ninterp] = i-1
    flag[-1] = values.shape[0]-1
    return values_int, flag

def interp_and_smooth(coords,ninterp_init=5,npad=10):

    # Interpolate and smooth
    points,flag = linear_fill(coords,ninterp=ninterp_init)
    weights = flag*0. + 0.1
    weights[flag>=0] = 1.
   
    ninterp = points.shape[0]*npad
    flag_fine = np.zeros(ninterp) - 1
    uval = np.unique(flag[flag>=0])
    for i,v in enumerate(uval): flag_fine[i*npad*ninterp_init] = v
    flag_fine[flag_fine==uval[-1]] = -1
    flag_fine[-1] = uval[-1]

    # Spline interpolate    
    tck, u = interpolate.splprep([points[:,0],points[:,1],points[:,2]],w=weights, s=0)
    u_fine = np.linspace(0,1,ninterp)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    
    return np.vstack([x_fine,y_fine,z_fine]).T, flag_fine

def overlay_sine(coords,amplitude=1.,period=1.,w0=0.,tether_start=False,tether_end=False,nteth=3):
    
    npts = coords.shape[0]
    lengths = np.zeros(npts-1)
    total_length = 0.
    for i in range(1,npts):
        x0,x1 = coords[i-1],coords[i]
        lengths[i-1] = np.linalg.norm(x1-x0)
        #direction = (x1-x0)/length
        
    total_length = np.sum(lengths)
    npeak = total_length / period
    
    t = np.cumsum(lengths) / total_length #np.linspace(0.,1.,npts)
    t = np.concatenate([[0.],t])
    w = np.zeros(npts) + (total_length*np.pi/period) + w0
    y = amplitude*np.sin(w*t)
        
    points = np.zeros([npts,3])
    points[:,0] = np.linspace(0.,total_length,npts)
    points[:,1] = y
    
    if tether_start:
        points[:nteth,1] *= np.flip(np.exp(-np.linspace(0,2,nteth)))
    if tether_end:
        points[-nteth:,1] *= np.exp(-np.linspace(0,2,nteth))

    pts = coords.copy()
    for i in range(1,npts):
        # Original curve directions (local axis)
        x0,x1 = coords[i-1],coords[i]
        x0[2],x1[2] = 0.,0.
        length = lengths[i-1] #np.linalg.norm(x1-x0)
        if length>0.:
            direction = (x1-x0)/length
            
            # Angle between local axis and axis axis that sine was constructed on
            angle = np.arctan2( xaxis[0]*direction[1] - xaxis[1]*direction[0], xaxis[0]*direction[0] + xaxis[1]*direction[1] )
            R = np.eye(3)
            R[0,0],R[0,1],R[1,0],R[1,1] = np.cos(angle),-np.sin(angle),np.sin(angle),np.cos(angle)
            
            # Sine component direction
            ref = points[i-1]
            ref[1] = 0.
            prot = R.dot(points[i]-ref) #/np.linalg.norm(points[i]))
            pts[i] = x1 + prot #/float(npts) #+ x1
            pts[i,2] = coords[i,2]             
    
    if tether_start:
        pts[0] = coords[0]
    if tether_end:
        pts[-1] = coords[-1]
    
    return  pts,w[-1]
    
def apply_to_graph(fname,graph=None,interp=True,ofile=None,interpolation_resolution=10.,interp_radius_factor=None,afH=None,afL=None,pfH=None,pfL=None,filter=None):

    if graph is None:
        graph = spatialgraph.SpatialGraph()
        print('Reading graph...')
        graph.read(fname)
        print('Graph read')
    
    if interp:
        ed = spatialgraph.Editor()
        # Don't remove intermediate nodes! Messes up ordering data from crawl algorithm!
        #print('Removing intermediate nodes')
        #graph = ed.remove_intermediate_nodes(graph)    
        # INTERPOLATE VESSELS  
        print(f'Interpolating graph ({interpolation_resolution} um)')
        ed.interpolate_edges(graph,interp_resolution=interpolation_resolution,interp_radius_factor=interp_radius_factor,filter=filter,noise_sd=0.)
    
    nodes = graph.get_data("VertexCoordinates")
    edgeconn = graph.get_data("EdgeConnectivity")
    nEdgePoint = graph.get_data('NumEdgePoints')
    radName = graph.get_radius_field()['name']
    points = graph.get_data('EdgePointCoordinates')
    radii = graph.get_data(radName)
    #edge_radii = graph.point_scalars_to_edge_scalars(name=radName)
    branch = graph.get_data('Branch').astype('int')
    order = graph.get_data('Order').astype('int')
    vtype = graph.get_data('VesselType').astype('int')
    nedge = edgeconn.shape[0]
    nnode = nodes.shape[0]
    npts = points.shape[0]
    epi = graph.edgepoint_edge_indices()
    
    if filter is None:
        filter = np.ones(edgeconn.shape[0],dtype='bool')
    
    # Mark start and end of each edge
    start_point = np.concatenate([[True],(epi[1:]!=epi[:-1])])
    strt = np.where(start_point)
    end_point = np.zeros(epi.shape[0],dtype='bool')
    end_point[strt[0]-1] = True

    new_points = points.copy()
    
    edges_corrected = np.zeros(nedge,dtype='bool')
    node_corrected = np.zeros(nnode,dtype='bool')
    
    # Ideally overlay by branch, allowing deviations to pass through bifurcations uninterrupted
    if branch is not None:
        print(f'Overlaying wriggle (by branch). afH={afH},afL={afL},pfH={pfH},pfL={pfL}')
        inlet,outlet = identify_inlet_outlet(graph)
        
        # Locate inlets/outlets to ignore them during processing
        inlet_edge = graph.get_edges_containing_node(inlet)
        outlet_edge = graph.get_edges_containing_node(outlet)
        branch[epi==inlet_edge] = -1
        branch[epi==outlet_edge] = -1
        # Also ignore first branch from inlet edges
        end_nodes,es = graph.connected_nodes(inlet)
        inlet_edge1 = arr([graph.get_edges_containing_node(x) for x in end_nodes]).flatten()
        for e in inlet_edge1:
            branch[epi==e] = -1
        end_nodes,es = graph.connected_nodes(outlet)
        outlet_edge1 = arr([graph.get_edges_containing_node(x) for x in end_nodes]).flatten()
        for e in outlet_edge1:
            branch[epi==e] = -1

        bvals = np.unique(branch)
        
        # Loop through branches in size order
        branch_max_radius = np.zeros(bvals.shape[0]) - 1.
        for i,bind in enumerate(tqdm(bvals)):
            if bind>=0:
                branch_max_radius[i] = np.max(radii[branch==bind])
        srt = np.argsort(branch_max_radius)
        bvals = np.flip(bvals[srt])
        branch_max_radius = np.flip(branch_max_radius[srt])
        #breakpoint()
        
        # Loop through branch indices in reverse size order
        for i,bind in enumerate(tqdm(bvals)):
            #print(branch_max_radius[i])
            # Indices of edgepoints from the current branch
            if bind>=0:
                inds = np.where(branch==bind)
            else:
                inds = [[]]
            
            if len(inds[0])>1:
                # Ordering of edges from current branch
                order0 = order[inds]
                srt = np.argsort(order0,kind='stable')
               
                # Reverse sort (to undo sort when storing data)
                revsrt = np.argsort(srt,kind='stable')
                # Sort data by order
                branch0 = points[inds][srt]
                order0 = order0[srt]
                brads = radii[inds][srt] 
                btype = vtype[inds][srt]    
                edges = epi[inds][srt]
                end_pts = end_point[inds][srt]

                cur_edges = np.unique(edges) #arr([x for j,x in enumerate(edges[1:]) if edges[j-1]!=edges[j]])
                #breakpoint()
                # Connections in the current branch
                cur_conns = edgeconn[cur_edges]
                # Node indices in the current branch
                cur_node_inds = np.unique(cur_conns)
                # Node coordinates in the current branch
                cur_nodes = nodes[cur_node_inds]
                # Whether the node position has changed in previous iterations (if so, update the edge below)
                cur_nodes_fixed = node_corrected[cur_node_inds]
                if np.any(cur_nodes_fixed):
                    fixed_node_inds = cur_node_inds[cur_nodes_fixed==True]
                    edge_inds_to_fix = [cur_edges[k] for k,x in enumerate(cur_conns) if np.any(np.in1d(fixed_node_inds,x))]
                    # Create a straight line between affected nodes
                    for k,eitf in enumerate(edge_inds_to_fix):
                        sind = np.where(epi==eitf)
                        npts = len(sind[0])
                        n0,n1 = nodes[edgeconn[eitf]]
                        line = np.linspace(n0,n1,npts)
                        points[sind] = line
                    # Re-assign branch coordinates to reflect changes
                    branch0 = points[inds][srt]
                else:
                    edge_inds_to_fix = [-1]

                # Indices of edges in current branch
                edgeinds = edges[np.sort(np.unique(edges,return_index=True)[1],kind='stable')]
                
                # Array to store which edges are reversed
                edgereverse = np.zeros(edges.shape[0],dtype='bool')
                
                # Create a copy of the original branch array
                branch0_init = branch0.copy()
                
                # Sort out the ordering of the branch edges
                if edgeinds.shape[0]>1:
                    # See if first edge transition is reversed or not
                    ec0,ec1 = edgeconn[edgeinds[0]],edgeconn[edgeinds[1]]
                    if ec0[0] in ec1:
                        edgereverse[0] = True
                        sinds = np.where(edges==edges[0])
                        branch0[sinds] = np.flip(branch0[sinds],axis=0)

                    count = 0
                    for j,e in enumerate(edges):
                        if j>0:
                            ec0,ec1 = edgeconn[edges[j-1]],edgeconn[edges[j]]
                            # If the edgepoint is associated with a different edge, see if it needs to be reverse to align with the previous edge
                            if not np.all(ec0==ec1):
                                # [0,1] -> [1,2]
                                if ec0[1]==ec1[0] and edgereverse[j-1]==False:
                                    rev = False
                                # [0,1] -> [2,1]
                                elif ec0[1]==ec1[1] and edgereverse[j-1]==False:
                                    rev = True
                                # [1,0] -> [1,2]
                                elif ec0[0]==ec1[0] and edgereverse[j-1]==True:
                                    rev = False
                                # [1,0] -> [2,1]
                                elif ec0[0]==ec1[1] and edgereverse[j-1]==True:
                                    rev = True
                                else:
                                    print('UNDEF:',count,j,ec0,ec1,edgereverse[j-1])
                                    breakpoint()
                                 
                                edgereverse[j] = rev
                                count += 1
                                #print(count,j,ec0,ec1,edgereverse[j-1],rev)
                                
                                sinds = np.where(edges==e)
                                sindspre = np.where(edges==edges[j-1])
                                                    
                                # Check nodes match!
                                n0 = nodes[ec1]
                                err = False
                                if not (np.all(n0[0]==branch0[sinds][0]) or not np.all(n0[1]==branch0[sinds][-1])) and \
                                   not (np.all(n0[1]==branch0[sinds][0]) or not np.all(n0[0]==branch0[sinds][-1])):
                                        err = True

                                if err:
                                    for k in range(0,j): print(edges[k],order0[k],branch0[k])
                                    print('***')
                                    for k in range(j-1,j+20): print(edges[k],order0[k],branch0[k])
                                    breakpoint()

                                if rev:
                                    branch0[sinds] = np.flip(branch0[sinds],axis=0)
                                
                                #plt.plot(branch0[sinds][:,0],branch0[sinds][:,1])
                                if not np.all(branch0[sinds][0]==branch0[sindspre][-1]):
                                    breakpoint()
                                
                            # If this and the previous edgepoint are from the same edge, copy the previous value over
                            else:
                                edgereverse[j] = edgereverse[j-1]
                        
                # Calculate segment lengths
                lengths = arr([np.linalg.norm(branch0[j]-branch0[j-1]) for j in range(1,branch0.shape[0])])
                # Remove doubled-up edge start/end points
                pnt_inds = np.concatenate([np.where(end_pts==False)[0],[len(end_pts)-1]])
                excluded_inds = np.where(~np.in1d(np.linspace(0,len(end_pts)-1,len(end_pts)),pnt_inds))
                
                # Removed 'doubled up' edgepoints
                branch0_int = branch0[pnt_inds]
                brads = brads[pnt_inds]

                if np.mean(radii[inds])>0.: # and np.sum(lengths)>period/2.:
                    # Smooth
                    if len(inds[0])>5: 
                        weights = np.ones(branch0_int.shape[0])
                        tck, u = interpolate.splprep([branch0_int[:,0],branch0_int[:,1],branch0_int[:,2]],w=weights, s=0)
                        u_fine = np.linspace(0,1,branch0_int.shape[0])
                        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
                        branch0_int[:,0],branch0_int[:,1] = x_fine,y_fine

                        # High frequency
                        if btype[0]==0: # artery
                            if afH is None:
                                afH = np.random.uniform(1.,3.5)
                            amplitude = np.clip(brads*afH,10.,None)        
                        elif btype[0]==1: # vein
                            if afL is None:
                                afL = np.random.uniform(1.,7.5)
                            amplitude = np.clip(brads*afL,10.,None)    
                        if pfH is None:    
                            pfH = np.random.uniform(15.,25)
                        period = brads*pfH #np.clip(brads*pfH,350.,None)
                        period[brads>30.] = 1000.
                        branch0_sin,w1 = overlay_sine(branch0_int,amplitude=amplitude*0.7,period=period,tether_start=False,tether_end=False)
                        
                        # Low frequency
                        amplitude = np.clip(brads*2.5,25.,None)        
                        if pfL is None:
                            pfL = np.random.uniform(30.,50.)
                        period = brads*pfL #np.clip(brads*pfL,350.,None)
                        branch0_sin,w1 = overlay_sine(branch0_sin,amplitude=amplitude*0.3,period=period,tether_start=False,tether_end=False)
                    else:
                        branch0_sin = branch0_int.copy()
                        
                    # Add in edge end points
                    branch0_sin_exp = np.zeros([len(inds[0]),3])
                    branch0_sin_exp[pnt_inds] = branch0_sin
                    if len(excluded_inds[0])>0:
                        branch0_sin_exp[excluded_inds] = branch0_sin_exp[excluded_inds[0]+1]
                        
                    # Set final coordinate to original values
                    #branch0_sin_exp[:,2] = branch0[:,2]
                    
                    if np.any(~np.isfinite(branch0_sin_exp)):
                        breakpoint()
                    if branch0_sin_exp.shape[0]!=branch0.shape[0]:
                        breakpoint()
                        
                    #breakpoint()
                    
                    if np.any(edgereverse):
                        for e in edgeinds:
                            sind = np.where(edges==e)
                            if edgereverse[sind][0]:
                                branch0_sin_exp[sind] = np.flip(branch0_sin_exp[sind],axis=0)
                                      
                    new_points[inds[0]] = branch0_sin_exp[revsrt]
                    
                    # Update node coords
                    if True:
                        # Connections constituting current branch
                        cur_edges = arr([x for j,x in enumerate(edges[1:]) if edges[j-1]!=edges[j]])
                        if cur_edges.shape[0]>0:
                            cur_conns = edgeconn[cur_edges]
                            
                            # Loop through branch nodes 
                            for j,conn in enumerate(cur_conns):
                                # Set the node coordinate to the new location
                                cpts = branch0_sin_exp[edges==cur_edges[j]]
                                # Get edges connected to current node
                                for k in [0,1]:
                                    ce = graph.get_edges_containing_node(conn[k])
                                    
                                    if not node_corrected[conn[k]]:                                
                                        #ce_rads = edge_radii[ce]
                                        ## Loop through each edge connected to current node
                                        #max_rad_ind = np.argmax(ce_rads)
                                        if k==0:
                                            nodes[conn[k]] = cpts[0]
                                        else:
                                            nodes[conn[k]] = cpts[-1]
                                        node_corrected[conn[k]] = True
                                    
                                    if False:        
                                        for e in ce:
                                        # Find edges not contained in the current branch
                                        #if  e not in edges: # and e not in edges_corrected:
                                            # Find edgepoints relating to the current edge index
                                            sind = np.where(epi==e)
                                            if edgeconn[e,0]==conn[k]:
                                                new_points[sind[0][0]] = nodes[conn[k]]
                                            elif edgeconn[e,1]==conn[k]:
                                                new_points[sind[0][-1]] = nodes[conn[k]]
                                            else:
                                                breakpoint()
                                            edges_corrected[e] = True
                    
                    #breakpoint()
            #break
    else:    
        print('Overlaying wriggle (by segment)...')
        
        for i,ec in enumerate(tqdm(edgeconn)):
            inds = np.where(epi==i)
            if len(inds[0])>5:

                branch0 = points[inds] # np.vstack([x0,x1,x2])
                brads = radii[inds]

                length = np.linalg.norm(branch0[-1]-branch0[0])

                if np.mean(brads)>60.:
                    period = length * 2.
                    amplitude = np.mean(brads) * 10.    
                else:
                    period = length / 5.
                    amplitude = np.mean(brads) * 10.
                
                branch0_int = branch0
                branch0_sin,w1 = overlay_sine(branch0_int,amplitude=amplitude,period=period,tether_start=True,tether_end=True)
                
                new_points[inds[0]] = branch0_sin
                
                for j,scalar in enumerate(scalars):
                    x0 = np.sum(nEdgePoint[:i])
                    n = nEdgePoint[i]
                    cur_data = scalar['data'][x0:x0+n]
                    new_scalar_data[j,inds] = np.linspace(cur_data[0],cur_data[1],len(inds[0]))
                
                if False and np.all(branch0[:,0]>bboxx[0]) and np.all(branch0[:,0]<bboxx[1]) and np.all(branch0[:,1]>bboxy[0]) and np.all(branch0[:,1]<bboxy[1]):
                    plt.plot(branch0_sin[:,0],branch0_sin[:,1],c='g')                    
    
    graph.set_data(nodes,name='VertexCoordinates')
    graph.set_data(new_points,name='EdgePointCoordinates')
    #for j,scalar in enumerate(scalars):
    #    graph.set_data(new_scalar_data[j],name=scalar['name'])
        
    graph.set_definition_size('POINT',new_points.shape[0])
    graph.set_graph_sizes()
    
    # Correct any edges that have not been adjusted to new node positions
    if True:
        nodes = graph.get_data("VertexCoordinates")
        edgeconn = graph.get_data("EdgeConnectivity")
        nEdgePoint = graph.get_data('NumEdgePoints')
        edgepoints = graph.get_data('EdgePointCoordinates')
        for i in range(graph.nedge):
            n0,n1 = nodes[edgeconn[i]]
            i0 = np.sum(nEdgePoint[:i])
            i1 = i0+nEdgePoint[i]
            epc = edgepoints[i0:i1]
            if not np.all(n0==epc[0]):
                #print(f"N0: {n0}, epc[0]:  {epc[0]}")
                edgepoints[i0] = n0
            if not np.all(n1==epc[-1]):
                #print(f"N1: {n1}, epc[-1]: {epc[-1]}")
                edgepoints[i1-1] = n1

        graph.set_data(edgepoints,name='EdgePointCoordinates')                

    if False:
        vtype = graph.point_scalars_to_edge_scalars(name='VesselType')
        edge_filter = np.zeros(vtype.shape[0],dtype='bool')
        if True: # artery only
            edge_filter[vtype==0] = True
        else:  #vein
            edge_filter[vtype==1] = True
        graph.plot_graph(edge_filter=edge_filter,scalar_color_name='Branch') #,scalar_color_name='Branch',cmap_range=[0,1])

    if ofile is not None:
        graph.write(ofile)
        
    return graph

