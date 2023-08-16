import numpy as np
from pymira import spatialgraph
from retinasim.utility import identify_inlet_outlet
from retinasim.eye import Eye
from tqdm import tqdm, trange
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import pickle
import os
join = os.path.join
arr = np.asarray

from retinasim.utility import make_dir

def impulse(t,delay,length=1.,height=1.):

    conc = np.zeros(t.shape[0])
    conc[t>=delay] = height
    conc[t>(delay+length)] = 0.
    return conc
    
def impulse_and_decay(t,delay,length=1.,height=1.,a=0.01):

    conc = np.zeros(t.shape[0])
    conc[t>=delay] = height
    decay = height*np.exp(-a*(t-delay-length))
    conc[t>(delay+length)] = decay[t>(delay+length)]
    return conc
    
def gaussian(t,delay,sd=5.,a=1.):

    return a*np.exp(-np.square(t-delay)/2*np.square(sd))
    
def bolus(  t,delay,
            a1 = 0.833,          # first pass scale
            a2 = 0.336,          # second pass scale
            t1 = 0.171,          # (min) first pass peak offset
            t2 = 0.364,          # (min) second passs peak offset
            sigma1 = 0.055,      # (min) first pass width
            sigma2 = 0.134,      # (min) second pass width
            alpha = 1.064,       # washout scale
            beta = 0.166,        # (/min) washout rate
            s = 37.772, 
            tau = 0.482):         # (min) washout offset
         
    # Time in minutes
    tMin = (t-delay) / 60.
    
    # First pass peak
    r1 = (a1 / (sigma1*np.sqrt(2.*np.pi))) * np.exp(-np.square(tMin-t1)/(2.*np.square(sigma1)))
    # Second pass peak
    r2 = (a2 / (sigma2*np.sqrt(2.*np.pi))) * np.exp(-np.square(tMin-t2)/(2.*np.square(sigma2)))
    # Washout
    wo = alpha * np.exp(-beta*(tMin-tau)) #/ (1. + np.exp(-s*(tMin-tau)))
    #wo[tMin<(delay/60.)] = 0.
    
    # Total conc
    conc = r1 + r2 + wo
    
    # Regularise...
    conc[tMin<0] = 0.
    conc[conc<0] = 0.
    conc[~np.isfinite(conc)] = 0.
        
    return conc    
    
def fix_zero_pd(dp,fl,Qin,Qout,fin,fout):

    """
    Sort out pressure drops=0 by assigining flow directions that maintain mass conservation
    """

    fnopd = np.where(dp==0.)[0]
    nopdflow = fl[fnopd]
    from itertools import product
    mnState = None
    mn = 1e6
    for i,state in enumerate(product([-1,1], repeat=len(fnopd))): 
        prod = np.sum(Qin)-np.sum(Qout)-np.sum([v*s for (v,s) in zip(nopdflow,state)])
        if prod<mn:
            mnState = arr(state)
            mn = prod
    ins = np.where(mnState>0)
    if len(ins)>0:
        fin = (np.append(fin[0],fnopd[ins[0]]),)
        Qin = np.sum(np.abs(fl[fin]))
    outs = np.where(mnState<0)
    if len(outs)>0:
        fout = (np.append(fout[0],fnopd[outs[0]]),)
        Qout = np.sum(np.abs(fl[fout]))
        
    return fin,fout,Qin,Qout

def crawl(graph=None,name='images',fname=None,proj_path=None,conc_dir=None,image_dir=None,calculate_conc=True,check_mass_cons=False,dt=None,duration=None,plot=True,downstream=True,    
           arteries_only=False,veins_only=False,ofilename=None,first_branch_index=0,ghost_edges=[],adhoc_pressure=False,write=True,store_conc=True,inlet=None,outlet=None):

    #f = os.path.join(dir_,fname)
    #proj_dir = join(dir_,name)
    if proj_path is not None:
        if not os.path.exists(proj_path):
            os.mkdir(proj_path)
        net_dir = join(proj_path,'network')
        if not os.path.exists(net_dir):
            os.mkdir(net_dir)
            
        crawl_dir = join(proj_path,'crawl')
        make_dir(crawl_dir,overwrite_existing=True)
    else: 
        net_dir = None
     
    #vis = None
        
    if calculate_conc:
        if conc_dir is None:
            conc_dir = 'concentration'
        if image_dir is None:
            image_dir = join(proj_path,conc_dir)
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
    else:
        image_dir = None

    # Plot window size        
    win_width = 1920*3
    win_height = win_width #1080*3
    
    # Read graph and interpolate
    if graph is None:
        graph = spatialgraph.SpatialGraph()
        print('Reading graph...')
        graph.read(join(proj_path,fname))
        print('Graph read')
    
    flow = graph.get_data('Flow')
    if flow is not None:
        print(f'Max. flow = {np.max(np.abs(flow))}')
    
    print('Calculating lengths...')
    lengths = graph.get_edge_lengths()
    
    nodes = graph.get_data("VertexCoordinates")
    edgeconn = graph.get_data("EdgeConnectivity")
    nEdgePoint = graph.get_data('NumEdgePoints')
    edgepoints = graph.get_data('EdgePointCoordinates')
    radName = graph.get_radius_field()['name']
    #print('Fetching edge flow...')
    edge_flow = graph.point_scalars_to_edge_scalars(name='Flow')
    if edge_flow is None:
        edge_flow = np.ones(edgeconn.shape[0])
    print('Fetching edge radii...')
    edge_radii = graph.point_scalars_to_edge_scalars(name=radName)
    print('Fetching node pressure...')
    #edge_pressure = graph.get_data(name='Pressure')
    node_pressure = graph.get_data(name='NodePressure')
    if node_pressure is None or adhoc_pressure:
        node_pressure = np.zeros(nodes.shape[0])
        adhoc_pressure = True
        downstream = True
    else:
        adhoc_pressure = False
    
    print('Fetching vessel type...')
    edge_vtype = graph.point_scalars_to_edge_scalars(name='VesselType')
    
    nnode = graph.nnode
    nedge = graph.nedge
    npoints = graph.nedgepoint
    gc = graph.get_node_count()
    
    # Edge volumes in um3
    volumes = lengths * np.square(edge_radii) * np.pi
    
    # To convert from (nL/min) to (um^3/s) use conversion below
    fluxConversion = 1e6 #/60. #1e6/60.
    # Velocity in um/s
    edge_velocity = np.abs(edge_flow)*fluxConversion / (np.square(edge_radii)*np.pi)
    
    if inlet is None or outlet is None:
        inlet,outlet = identify_inlet_outlet(graph)
    
    # If crawling upstream, swap inlets and outlets
    if not downstream:
        outlet,inlet = inlet,outlet
    elif adhoc_pressure and inlet is None and outlet is not None: # if only veins and no pressure estimates...
        outlet,inlet = inlet,outlet

    # Slow down capillary flow by making vessels longer (useful for networks with directly connected arteries and veins)
    if True:
        s_cap = np.where(edge_vtype==2)
        cap_f = 3. # s
        lengths[s_cap] *= cap_f
        edge_velocity[s_cap] /= cap_f
    
    # Check mass conservation
    if check_mass_cons and not adhoc_pressure:
        print('Checking mass conservation...')
        mass_conserved = np.ones(nnode,dtype='bool')
        edge_dp0 = np.zeros(nedge)
        fl_dif = np.zeros(nnode)
        gc = graph.get_node_count()
        for i in trange(nnode):
            edges = graph.get_edges_containing_node(i)
            end_nodes = arr([e[e!=i] for e in edgeconn[edges]]).flatten()
            
            # Pressure difference between current node and connected nodes
            dp = node_pressure[i] - node_pressure[end_nodes].flatten()
            # Flow in current edges
            if edge_flow is not None:
                fl = edge_flow[edges]
            else:
                fl = np.ones(edges.shape[0])
            # Vessel categories
            vt = edge_vtype[edges]
            
            # Identify upstream (pressure increase) and downstream (pressure decrease) nodes
            if downstream:
                fin_node = np.where(dp<0.)
                fout_node = np.where(dp>0.)
            else:
                fin_node = np.where(dp>0.)
                fout_node = np.where(dp<0.)            
            fzero = np.where(dp==0.)
            
            # Calculate upstream and downstream total flow
            Qin = np.sum(np.abs(fl[fin_node]))
            Qout = np.sum(np.abs(fl[fout_node]))
            
            if Qin!=Qout and gc[i]>1:
                #print(f'Qi!=Qo: {i}, {Qin-Qout}')
                mass_conserved[i] = False
                fl_dif[i] = np.abs(Qin-Qout)
                
            if len(fzero[0])>0 and gc[i]>1:
                #print(f'Dp=0: {edges[fzero]}')
                edge_dp0[edges[fzero]] = True
                
            # Test connectivity
            dp_pol = np.sort(dp / np.abs(dp)) # pressure polarity
            if np.all(vt==0) and (not dp_pol[0]==-1 or not np.all(dp_pol[1:]==1)):
                breakpoint()
            elif np.all(vt==1) and (not np.flip(dp_pol[0]==-1) and not np.all(np.flip(dp_pol[1:])==1)):     
                breakpoint()
            
        print(f'Mass conserved in {np.sum(mass_conserved)*100/nnode}% of nodes')
        print(f'Max. flow disparity: {np.max(fl_dif)}')
        if np.any(fl_dif>1e-3):
            breakpoint()
    
    # Initialise variables
    crawl_progress = np.zeros(nedge,dtype='int')
    delay = np.zeros(nedge)
    earliest_arrival = np.zeros(nedge) - 1.
    edge_order = np.zeros(nedge,dtype='int') - 1
    edge_visited = np.zeros(nedge,dtype='int') #,dtype='bool')
    edge_ghost = np.zeros(nedge,dtype='int') - 1 #,dtype='bool')
    node_visited = np.zeros(nnode,dtype='bool')
    node_processed = np.zeros(nnode,dtype='bool')

    # Initialise
    edges = graph.get_edges_containing_node(inlet)
    
    if calculate_conc:
    
        # Initialise time
        if dt is None:
            dt = 0.1 #1./30. #s (30 fps?)
        if duration is None:
            duration = 25.*60. #s
        nt = int(np.ceil(duration/dt))
        time = np.linspace(0.,duration,nt)
    
        # Initialise concentration variable
        conc = np.zeros([nedge,nt])
        
        if False:
            impulse_length = 5. #s dt*nt #dt*5
            conc[edges] = impulse(time,0.,length=impulse_length)
        elif False:
            conc[edges] = impulse_and_decay(time,0.,length=180.,a=0.01) #impulse(time,0.,length=impulse_length)
        else:
            injection_duration = 10. # s
            halflife = 23.5 #286. # min
            prk = bolus(time,0.,sigma1=injection_duration/60.,sigma2=2.5*injection_duration/60.,beta=1./halflife)
            conc[edges] = prk / np.max(prk)
        
    crawl_progress[edges] = 1
    edge_visited[edges] = 1 #True
    for e in edges:
        if e in ghost_edges:
            edge_ghost[edges] = 1 #True
        else:
            edge_ghost[edges] = 0 #True
    node_visited[inlet] = True
    end_nodes = edgeconn[edges].flatten()
    end_nodes = end_nodes[end_nodes!=inlet]
    
    # If no pressure values provided, add them in to create a false pressure drop for all connected nodes
    if adhoc_pressure:
        node_pressure[inlet] = 1.
    dp = node_pressure[inlet] - node_pressure[end_nodes].flatten()
    
    #branch_order = np.zeros(nedge,dtype='int') - 1
    branch_index = np.zeros(nedge,dtype='int') - 1
    # Assign the first branch index (usually zero)
    branch_index[edges] = first_branch_index

    pbar = tqdm(total=nedge,desc='Crawling')
    cur_nodes = arr(end_nodes)
    cur_nodes_held = [0 for _ in range(len(cur_nodes))]
    cur_node_propogated_delays = np.zeros(len(end_nodes))
    cur_node_ghost = np.zeros(len(end_nodes),dtype='int')
    count = 0
    itcount = 0
    
    hold_limit = 1000
    
    cur_branch_order = 0
    #breakpoint()
    while True:
        nedgeupdate = 0
        next_nodes = []
        next_node_propogated_delay = []
        nodes_held = []
        next_nodes_ghost = []
        for i,cur_node in enumerate(cur_nodes):
        
            # Select current propogated delay
            cur_node_propogated_delay = cur_node_propogated_delays[i]
        
            # Edge indices connected to current node
            edges = graph.get_edges_containing_node(cur_node)
            
            # Filter for arteries only
            if arteries_only and np.any(edge_vtype[edges]!=0):
                edges = edges[edge_vtype[edges]==0]
            elif veins_only and np.any(edge_vtype[edges]!=1):
                edges = edges[edge_vtype[edges]==1]
            
            # Find the nodes at the end of each connected edge by removing the current (source) node from the end node list
            end_nodes = arr([e[e!=cur_node] for e in edgeconn[edges]]).flatten()
            
            if len(end_nodes)>0:
            
                # Pressure difference between current node and connected nodes
                if adhoc_pressure:
                    node_pressure[cur_node] = 1.
                dp = node_pressure[cur_node] - node_pressure[end_nodes].flatten()
                # Flow in current edges
                fl = edge_flow[edges]
                # Vessel categories
                vt = edge_vtype[edges]
                # Radii
                rs = edge_radii[edges]
                # Volumes
                vl = volumes[edges]
                # Ghost
                gh = edge_ghost[edges]
                    
                # Identify upstream (pressure increase) and downstream (pressure decrease) nodes
                if downstream:
                    if adhoc_pressure==False:
                        fin_node = np.where(dp<0.)
                    else:
                        fin_node = np.where(dp<=0.)
                    fout_node = np.where(dp>0.)
                else:
                    if adhoc_pressure==False:
                        fin_node = np.where(dp>0.)
                    else:
                        fin_node = np.where(dp>=0.)
                    fout_node = np.where(dp<0.)
                
                # Calculate upstream and downstream total flow
                Qin = np.sum(np.abs(fl[fin_node]))
                Qout = np.sum(np.abs(fl[fout_node]))
                
                invalid = False
                filter_conns = False
                if filter_conns:
                    dp_pol = np.sort(dp / np.abs(dp)) # pressure polarity
                    if np.all(vt==0) and (not dp_pol[0]==-1 or not np.all(dp_pol[1:]==1)):
                        invalid = True
                    elif np.all(vt==1) and (not np.flip(dp_pol[0]==-1) and not np.all(np.flip(dp_pol[1:])==1)):     
                        invalid = True

                # Check if upstream held nodes have expired - if so, allow flow if at least one inlet has been visited (that isn't held)
                #print(len(cur_nodes_held),end_nodes[fin_node])
                inlet_held_time = arr([cur_nodes_held[k] if x in cur_nodes else 0. for k,x in enumerate(end_nodes[fin_node]) ])               
                # If inlet held longer than limit but edge hasn't been visited, and
                # at least one inlet not held and has been visited
                if np.all(edge_visited[edges[fin_node]]>0.):
                #if np.all( (inlet_held_time>=hold_limit) | (edge_visited[edges[fin_node]]>0.) ):
                    invalid = False
                    
                # If not all upstream nodes have been visited, hold the node for a future iteration
                else:
                    # If held count equals -1, then make node valid            
                    if cur_nodes_held[i]>=0:
                        invalid = True
                    
                        # If the node has already been added, add it automatically into the next iteration
                        if cur_node in next_nodes:
                            nodes_held = [nodes_held[k] if x!=cur_node else 1 for k,x in enumerate(next_nodes)]
                        else:
                            next_nodes.extend([cur_node])
                            nodes_held.extend([cur_nodes_held[i]+1])
                            next_node_propogated_delay.extend([cur_node_propogated_delay])
                            next_nodes_ghost.extend([cur_node_ghost[i]])
                        
                # Otherwise, process the node
                if not invalid and len(fout_node[0])>0:

                    # Calculate delay caused by transit through edges
                    delays = arr(cur_node_propogated_delay + lengths[edges[fout_node]]/edge_velocity[edges[fout_node]])
                    if len(fin_node[0])>0:
                        delay[edges[fout_node]] = np.min(delay[edges[fin_node]]) + lengths[edges[fout_node]]/edge_velocity[edges[fout_node]]
                    propogated_delay = arr(cur_node_propogated_delay + lengths[edges[fout_node]]/edge_velocity[edges[fout_node]])
                    
                    # Calculate concentration in downstream edges
                    for j,edge in enumerate(edges):
                        valid_pressure = False
                        if downstream and dp[j]>0.:
                            valid_pressure = True
                        elif not downstream and dp[j]<0.:
                            valid_pressure = True
                            
                        if not edge_visited[edges[j]] and valid_pressure: # x
                            if calculate_conc:
                                # Functional form
                                if False:
                                    conc[edge,:] += np.abs(fl[j])*concFunc(time,cur_node_propogated_delay,length=impulse_length)/Qin
                                # Using input data
                                else:
                                    #conc_in = arr([a*b for a,b in zip(conc[edges[fin_node[0]]],vl[fin_node[0]])])
                                    conc_in = conc[edges[fin_node[0]]]
                                    conc_in = np.sum(conc_in,axis=0).clip(0,1.)
                                    ignore_delay = False
                                    if ignore_delay:
                                        cur_conc = np.abs(fl[j])*conc_in/Qin
                                    else:
                                        f = interp1d(time, conc_in, fill_value='extrapolate')
                                        tp = np.linspace(0-cur_node_propogated_delay,(dt*nt)-cur_node_propogated_delay,nt)
                                        cur_conc = np.abs(fl[j])*f(tp)/(Qin) #*vl[j])
                                        cur_conc[time<cur_node_propogated_delay] = 0.
                                    conc[edge,:] += cur_conc
                            nedgeupdate += 1
                            
                            edge_visited[edges[j]] += 1
                                                                    
                            inlet_ghost = gh[fin_node]
                            if edges[j] in ghost_edges:
                                edge_ghost[edges[j]] = 1
                            elif np.all(inlet_ghost==1): # or cur_node_ghost[i]==1:
                                edge_ghost[edges[j]] = 1
                            else:
                                edge_ghost[edges[j]] = 0
                            #edge_ghost[edges[j]] = cur_node_ghost[i]
                            
                        edge_order[edges] = [count if e<0 else e for k,e in enumerate(edge_order[edges])]
                        crawl_progress[edges] = 1

                    # Assign nodes to be used in next iteration
                    for k,nn in enumerate(end_nodes[fout_node]):
                        if nn not in next_nodes:
                            next_nodes.extend([nn])
                            # Assign zero to held value (i.e. don't hold)
                            nodes_held.extend([0])
                            next_node_propogated_delay.extend([propogated_delay[k]])

                            if edges[k] in ghost_edges:
                                next_nodes_ghost.extend([1])
                            elif cur_node_ghost[i]==1:
                                next_nodes_ghost.extend([1])
                            else:
                                next_nodes_ghost.extend([0])
                    
                    node_processed[cur_node] = True
                    
                    # Continue branch along largest radius
                    # Identify edge with largest radius
                    try:
                        mxrad = np.argmax(rs[fout_node[0]])
                    except Exception as e:
                        print(e)
                        breakpoint()
                    # Set the largest downstream branch index to the smallest upstream branch index
                    branch_index[edges[fout_node[0][mxrad]]] = np.min(branch_index[edges[fin_node]])
                    # Find the current largest branch index
                    mxind = np.max(branch_index)
                    # Edit indices to remove the largest outlet
                    inds = fout_node[0].copy()
                    # Delete largest radius outlet index
                    inds = np.delete(inds,mxrad)
                    nval = len(inds)
                    if nval>0:
                        # Assign new branch indices to all other outlet edges
                        branch_index[edges[inds]] = np.linspace(mxind+1,mxind+1+nval,nval)
                        
                    count += 1
                
        node_visited[cur_node] = True
        
        # Assign values for next iteration
        cur_nodes = arr(next_nodes).flatten()
        cur_nodes_unq,unq_inds = np.unique(cur_nodes,return_index=True) 
        cur_nodes_held = arr(nodes_held).flatten()
        cur_node_propogated_delays = arr(next_node_propogated_delay).flatten() #[unq_inds]
        cur_node_ghost = arr(next_nodes_ghost).flatten()
        
        # Limit the time a node can be held for
        #s_val = np.where(cur_nodes_held<hold_limit)
        #s_inval = np.where(cur_nodes_held>=hold_limit)
        #if len(s_inval[0])>0:
        #    cur_nodes_held[s_inval[0]] = -1
        
        # Check whether next iteration values require exit from loop
        if len(cur_nodes)==0: 
            break

        if len(cur_node_propogated_delays)>0:
            min_delay = np.min(cur_node_propogated_delays)
        else:
            min_delay = -1.
        
        # Plot
        update = 50
        if False and (itcount%update==0):
            cmap_range = [0,1]
            min_radius = 0.
            grab_file = join(crawl_dir,f'crawl_{str(itcount).zfill(4)}.png')
            
            visited_edge_tmp = np.repeat(edge_visited,nEdgePoint).astype('int')
            visited_ghost = np.repeat(edge_ghost,nEdgePoint).astype('int')
            visited_edge_tmp[visited_edge_tmp>0] = 1
            #visited_edge_tmp[visited_ghost>0] = 0
            visited_edge_tmp[visited_edge_tmp<=0] = 0
            
            #if vis is None:
            vis = graph.plot_graph(edge_color=visited_edge_tmp,cmap='jet',cmap_range=cmap_range,show=False,block=False,win_width=win_width,win_height=win_height,radius_scale=1.,min_radius=min_radius,grab_file=grab_file)
            #else:
            vis.set_cylinder_colors(edge_color=visited_edge_tmp)
            vis.additional_meshes = None
            for k,n in enumerate(cur_nodes_held):
                if n>0:
                    centre = nodes[cur_nodes[k]]
                    vis.add_torus(centre=centre,torus_radius=200.,tube_radius=20.) 
            vis.screen_grab(grab_file)
            vis.destroy_window()
            print(grab_file)
            #edge_highlight = edges
            #cyls = graph.plot_graph(edge_color=crawl_progress,cmap='jet',cmap_range=cmap_range,bgcolor=[0,0,0],plot=True,grab=False,win_width=1920,win_height=1080,radius_scale=1.,min_radius=min_radius,edge_highlight=edge_highlight)
            #print(f'Minimum delay: {np.min(cur_node_propogated_delays)}')
        
        itcount += 1
        
        # Update progress bar
        pbar.n = len(np.where(edge_visited>0)[0]) 
        
        # Ensure node front value are valid
        #s_held = np.where((cur_nodes_held>0) & (node_processed[cur_nodes]==True))
        #if len(s_held[0])>0:
        #    keep = np.where(~(cur_nodes_held>0 ) | ~(node_processed[cur_nodes]==True))
        #    cur_nodes = cur_nodes[keep]
        #    cur_nodes_held = cur_nodes_held[keep]
        #    cur_node_propogated_delays = cur_node_propogated_delays[keep]
         
        # Measure size of front   
        s_held = np.where(cur_nodes_held>0)
        # Update progress bar description
        pbar.set_description(f"Crawling ({itcount}, {len(cur_nodes)}, {len(s_held[0])}, {np.max(cur_nodes_held)}, {nedgeupdate})")
        pbar.refresh()

        # If all edges have been visited, break the loop
        if np.all(edge_visited>0):
            break
        #if np.all(cur_nodes_held>hold_limit):
        #    break
        if nedgeupdate==0:
            sind = np.where((cur_nodes_held<hold_limit) & (cur_nodes_held>0))
            if len(sind[0])>0:
                #cur_nodes_held[sind] += hold_limit - np.max(cur_nodes_held[sind])
                pass
            else:
                #breakpoint()
                pass                
         
    # Destroy progress bar
    pbar.close()
    
    # Destory visualisation window
    #if vis is not None:
    #    vis.destroy_window()
    
    branch_index_edge = np.repeat(branch_index,nEdgePoint).astype('int')
    order_edge = np.repeat(edge_order,nEdgePoint).astype('int')
    visited_edge = np.repeat(edge_visited,nEdgePoint).astype('int')
    visited_ghost = np.repeat(edge_ghost,nEdgePoint).astype('int')
    visited_edge[visited_edge>0] = 1
    visited_edge[visited_ghost>0] = 0
    visited_edge[visited_edge<=0] = 0
    perfused = visited_ghost.copy() #np.abs(1-visited_ghost)
    perfused[visited_ghost==0] = 1
    perfused[visited_ghost==1] = 0
    perfused[visited_ghost==-1] = 0

    if plot and net_dir is not None:
        # Artery-vein image   
        cmap_range = [None,None]
        min_radius = 0.
        grab_file = join(net_dir,f'artery_vein.png')
        print('Creating artery-vein graph')
        tp = graph.plot_graph(scalar_color_name='VesselType',cmap='jet',cmap_range=cmap_range,show=False,win_width=win_width,win_height=win_height,block=False)
        tp.screen_grab(grab_file)
    
    # Perfused
    pname = 'Perfused'
    if pname not in graph.fieldNames:
        graph.add_field(name=pname,definition='POINT',type='int',nelements=1,nentries=[npoints])
    graph.set_data(perfused,name=pname)
    if plot and net_dir is not None:
        cmap_range = [0,2]
        grab_file = join(net_dir,f'{pname}.png')
        print('Creating perfused graph')
        tp.set_cylinder_colors(scalar_color_name=pname,cmap='jet',cmap_range=cmap_range)
        tp.screen_grab(grab_file)
    
    # Order
    pname = 'Order'
    if pname not in graph.fieldNames:
        graph.add_field(name=pname,definition='POINT',type='int',nelements=1,nentries=[npoints])
    graph.set_data(order_edge,name=pname)
    if plot and net_dir is not None:
        cmap_range = [None,None]
        grab_file = join(net_dir,f'{pname}.png')
        print('Creating order graph')
        tp.set_cylinder_colors(scalar_color_name=pname,cmap='jet',cmap_range=cmap_range)
        tp.screen_grab(grab_file)
 
    # Branch
    pname = 'Branch'
    if pname not in graph.fieldNames:
        graph.add_field(name=pname,definition='POINT',type='int',nelements=1,nentries=[npoints])
    graph.set_data(branch_index_edge,name=pname)
    if plot and net_dir is not None:
        cmap_range = [0,500]
        grab_file = join(net_dir,f'{pname}.png')
        print('Creating branch graph')
        tp.set_cylinder_colors(scalar_color_name=pname,cmap='jet',cmap_range=cmap_range)
        tp.screen_grab(grab_file)

    # Delay
    if False:
        pname = 'Delay'
        if pname not in graph.fieldNames:
            graph.add_field(name=pname,definition='POINT',type='float',nelements=1,nentries=[npoints])
        delay_edge = np.repeat(delay,nEdgePoint)
        graph.set_data(delay_edge,name=pname)
        if plot and net_dir is not None:
            cmap_range = [0.,10.]
            grab_file = join(net_dir,f'{pname}.png')
            print('Creating delay graph')
            tp.set_cylinder_colors(scalar_color_name=pname,cmap='jet',cmap_range=cmap_range)
            tp.screen_grab(grab_file)
    
    if plot and net_dir is not None:
        tp.destroy_window()
    
    if calculate_conc:
        if store_conc:
            print('Storing injection result...')
            #skip = 1
            #conc = conc[:,::skip]
            #time = time[::skip]
            pfile = join(image_dir,"concentration_data.p")
            with open(pfile, 'wb') as handle:
                pickle.dump([conc,time], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        #if plot:
        #    conc_movie(graph,data=conc,time=time,odir=image_dir,win_width=win_width,win_height=win_width,skip=1)
        return graph, conc, time
    else:
        return graph

    if False: #add_conc:
        print('Adding concentration data to graph...')
        for i,t in enumerate(tqdm(time)):
            cur_conc = conc[:,i]
            cur_conc_edge = np.repeat(cur_conc,nEdgePoint)

            new_field = f'Concentration_{str(i).zfill(4)}'
            if new_field not in graph.fieldNames:
                graph.add_field(name=new_field,definition='POINT',type='float',nelements=1,nentries=[npoints])        
            graph.set_data(cur_conc_edge,name=new_field)

        ofile = os.path.join(proj_path,'concentration_infusion.am') 
        print(f'Writing graph with concentration data... {ofile}')   
        graph.write(ofile) 
    elif write:
        if ofilename is None:
            ofilename = 'retina_crawl_result.am'
        ofile = os.path.join(proj_path,ofilename) 
        print(f'Writing graph with extended crawl data... {ofile}')  
        #breakpoint() 
        graph.write(ofile)
        
    return graph
        
def conc_movie(graph,data=None,time=None,recon_times=None,odir=None,win_width=1920,win_height=1080, quick=False, skip=1, log_scale=False, eye=None, highlight_optic_disc=False, highlight_macula=False,movie_naming=True,cmap_range=[0,0.01],output_mesh=False):

    if data is None:
        cfields = [f for f in graph.fieldNames if 'Concentration' in f]
        nt = len(cfields)
    else:
        nt = data.shape[1]
        cfields = []
        
    if time is None:
        time = np.linspace(0,nt-1,nt,dtype='int')
        time_unit = 'frames'
    else:
        time_unit = 'seconds'

    if recon_times is not None:
        recon_time_ind = arr([np.argmin(np.abs(time-x)) for x in recon_times])
    else:
        recon_time_ind = np.linspace(0,nt-1,nt,dtype='int')
    nrecon = recon_time_ind.shape[0]

    #cmap_range = [0.,cmax] # 0.002
    min_radius = 0.
    
    if not os.path.exists(odir):
        os.mkdir(odir)
        
    scalars = graph.get_scalars()
    scalarNames = [x['name'] for x in scalars]
    
    nEdgePoint = graph.get_data('NumEdgePoints')

    show = False
    tp = None
    
    domain_type = 'cylinder'
    if True and eye is not None:
        domain = eye.domain
    else:
        #domain = arr([[-2000,18000],[-10000,10000],[-10000,10000]])
        domain = arr([[-20000,20000],[-20000,20000],[-10000,10000]])
        
    #dims = arr([128,128,1])
    #dx,dy,dz = (domain[:,1]-domain[:,0])/dims
    #leakage = np.zeros(dims)

    count = 0
    for i in tqdm(recon_time_ind): #range(0,nt,skip):
        ind = str(i).zfill(8)
        cname = f'Concentration_{ind}'
        
        if cname not in scalarNames and data is None:
            print(f'Could not locate {cname}')
            breakpoint()
        else:
        
            if time is not None:
                if time_unit=='seconds':
                    unit_text = 'min.'
                    nmin = int(np.floor(time[i]/60.))
                    nsec = int(np.floor(time[i]-nmin*60.))
                    nms = int((time[i]-nmin*60.-nsec)*1000)
                    text = f"Time = {nmin:02} min {nsec:02} sec {nms:03} msec"
                    fid = f"{nmin:02}_{nsec:02}_{nms:03}"
                else:
                    unit_text = time_unit
                    text = f"Time = {time[i]} {unit_text}"
                    fid = f"{i}"
            else:
                text = f"Time = {ind}"
                fid = f"{ind}"
        
            if movie_naming:
                fid = f"{str(count).zfill(8)}"
                
            grab_file = join(odir,f'concentration_{fid}.png')
            if tp is None:
                bgcolor = arr([0.,0.,0.]) # arr([0.2,0.2,0.2])
                tp = graph.plot_graph(show=False,block=False,bgcolor=bgcolor,min_radius=min_radius,domain_type=domain_type,domain=domain,win_width=win_width,win_height=win_height)
                
            if data is not None:
                cur_conc = data[:,i]
                cur_conc_edge = np.repeat(cur_conc,nEdgePoint)
                if log_scale:
                    cur_conc_edge[cur_conc_edge<=0.] = 1e-12
                    cur_conc_edge = np.log(cur_conc_edge)
                tp.set_cylinder_colors(edge_color=cur_conc_edge,cmap='gray',cmap_range=cmap_range,scalar_color_name='')
                
            else:    
                tp.set_cylinder_colors(scalar_color_name=cname,cmap='gray',cmap_range=cmap_range)
            
            if output_mesh:
                tp.update()
                gmesh = tp.cylinders_combined

                import open3d as o3d
                if count==0:
                    mesh_file = grab_file.replace('.png','.ply')
                    o3d.io.write_triangle_mesh(mesh_file,gmesh,write_vertex_colors=True)
                    mesh_vertex_colors_cur = np.asarray(gmesh.vertex_colors)
                    mesh_vertex_colors = np.zeros([recon_time_ind.shape[0],mesh_vertex_colors_cur.shape[0],3])
                mesh_vertex_colors_cur = np.asarray(gmesh.vertex_colors)
                mesh_vertex_colors[count] = mesh_vertex_colors_cur
            else: 
                if highlight_optic_disc and eye is not None:
                    centre = eye.optic_disc_centre
                    centre[2] = -20.
                    tp.add_torus(color=[1.,1.,1.,],centre=centre,torus_radius=eye.optic_disc_radius/2.,tube_radius=20.) # radial_resolution=30, tubular_resolution=20
                if highlight_macula and eye is not None:
                    centre = eye.fovea_centre
                    centre[2] = -20.
                    tp.add_cylinder(color=[0.,0.,0.],centre=centre,radius=eye.fovea_radius,height=20.)
                    
                tp.update()
                tp.screen_grab(grab_file)
                
                # Add text overlay
                if False:
                    from PIL import Image
                    from PIL import ImageDraw
                    from PIL import ImageFont
                    img = Image.open(grab_file)
                    I1 = ImageDraw.Draw(img)
                    #myFont = ImageFont.truetype('FreeMono.ttf', 65)
                    font_size = np.clip(int(18 * win_width/1080.),18,None)
                    font = ImageFont.truetype("FreeMono.ttf", font_size)  # /usr/share/fonts/truetype/freefont/
                    
                    I1.text((28, 36), text, fill=(255, 255, 255),font=font)
                    
                    img.save(grab_file)
                
                print(f'Written {grab_file}') 
            count += 1
            
    if output_mesh:
        ofile = join(odir,f'vertex_colors.npy')
        for i in range(656): 
            np.save(ofile.replace('.npy',f'{str(i).zfill(8)}.npy'),mesh_vertex_colors[i])
        #np.save(ofile,mesh_vertex_colors)
        print(f'Mesh colours written to {ofile}')
    
    tp.destroy_window()
    
def recon_conc(graph=None,path=None,gfile=None,geometry_file=None,image_dir=None,conc_dir=None,eye=None,skip=1,recon_times=None,movie_naming=True,delete_previous=True,win_width=1920,win_height=1080,highlight_optic_disc=False,cmax=0.01,log_scale=False,cmap_range=[0.,0.01]):

    #win_width = 1920*3
    #win_height = win_width #1080*3

    if path is None:
        return
    if gfile is None:
        gfile = join(path,'retina_cco_a2v_reanimate_crawl.am') 
    if conc_dir is None:
        conc_dir = join(path,'concentration')
    pfile = join(conc_dir,"concentration_data.p")

    if eye is None:    
        eye = Eye()
        eye.load(geometry_file)
    
    if delete_previous:
        import os, shutil
        for filename in os.listdir(conc_dir):
            if filename.endswith('.png'):
                file_path = os.path.join(conc_dir, filename)
                os.unlink(file_path)
           
    if graph is None: 
        graph = spatialgraph.SpatialGraph()
        print(f'Reading graph file: {gfile}')
        graph.read(gfile)
    
    with open(pfile,'rb') as handle:
        data = pickle.load(handle)
        conc,time = data
    
    conc_movie(graph,data=conc,time=time,recon_times=recon_times,odir=conc_dir,win_width=win_width,win_height=win_width,skip=skip,eye=eye,movie_naming=movie_naming,highlight_optic_disc=highlight_optic_disc,log_scale=log_scale,cmap_range=cmap_range) # cmax=cmax,
    
def plot_bolus():
 
    t = np.linspace(0.,3.*60.,1000)
    conc = bolus(t,0.)
    plt.plot(t/60.,conc)
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration (a.u.)')
    plt.ylim((0,8))
    fig = plt.gcf()
    fig.savefig(r'C:\Users\simon\Desktop\conc.png')
    plt.show()
    breakpoint()
    
if __name__=='__main__':

    #plot_bolus()

    if True:
        name = 'gaussian3'

        recon_only = True
        delete_previous = True
        duration = 10.*60. # seconds
        dt = 1. # seconds
        nimage = int(duration/0.1)
        recon_times = np.linspace(0.,duration,nimage)
        movie_naming = True
        
        path = r'/OUTPUT/PATH/'
        gfile = 'GRAPH_FILE.am'
        conc_dir = 'ffa'
        graph = None
        eye = Eye()

        if not recon_only:
            graph = crawl(name=name,proj_path=path,fname=gfile,calculate_conc=True,conc_dir=conc_dir)

        conc_dir = join(path,'ffa')
        cmap_range = [0.,0.05]
        recon_conc(graph=graph,gfile=join(path,gfile),path=path,conc_dir=join(path,conc_dir),eye=eye,recon_times=recon_times,movie_naming=movie_naming,cmap_range=cmap_range)
    
