from pymira import spatialgraph
import numpy as np
import os
import nibabel as nib
from matplotlib import pyplot as plt
from PIL import ImageDraw, Image
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from tqdm import trange, tqdm
import open3d as o3d

from retinasim.eye import Eye
from retinasim import geometry

join = os.path.join
arr = np.asarray

"""
Create an OCT-A style enface image from a spatial graph (vessel network) file
"""

def inside_fov(coord,fov):

    if fov is None:
        return np.ones(coord.shape[0],dtype='bool')
        
    res = np.zeros(coord.shape[0],dtype='bool')
    stmp = np.where((coord[:,0]>=fov[0,0]) & (coord[:,0]<=fov[0,1]) & (coord[:,1]>=fov[1,0]) & (coord[:,1]<=fov[1,1]))
    res[stmp[0]] = True
    return res

def coord_to_index(coord,fov,dims):
    index = (((coord - fov[:,0]) / (fov[:,1]-fov[:,0])) * dims).astype('int')
    return index
    
def enface_from_graph(graph,dims=[500,500],fov=None,radius_scale=1.,eye=None,overlay_features=False,fixed_radius=None,separate_artery_vein=False,exclude_capillaries=True):

    xdim,ydim = dims[0],dims[1]

    # Grab graph data
    nodecoords = graph.get_data('VertexCoordinates')
    conns = graph.get_data('EdgeConnectivity')
    nedgepoints = graph.get_data('NumEdgePoints')
    edgepoints = graph.get_data('EdgePointCoordinates')
    radName = graph.get_radius_field()['name']
    radii = graph.get_data(radName)
    vtype = graph.get_data('VesselType')
    
    # Create images for drawing vessels and storing enface images
    enface_seg_image_art = Image.new('I', (xdim, ydim), 0)  
    enface_seg_image_vein = Image.new('I', (xdim, ydim), 0)      
    enface_seg_image_draw_art = ImageDraw.Draw(enface_seg_image_art)
    enface_seg_image_draw_vein = ImageDraw.Draw(enface_seg_image_vein)
    
    if fov is None:
        fov = arr(graph.edge_spatial_extent())[:2]
    
    view_size = fov[:,1] - fov[:,0]
    pixsize = view_size / dims
    
    # Loop through each edge in the graph
    if fov is not None:
        bbox = [ arr([fov[0,0],fov[1,0]]), arr([fov[0,0],fov[1,1]]), arr([fov[0,1],fov[1,1]]), arr([fov[0,1],fov[1,0]]) ] # bottom left, top left, top right, bottom right
        bbox_edges = [ [bbox[0],bbox[1]], [bbox[1],bbox[2]], [bbox[2],bbox[3]], [bbox[3],bbox[0]] ]
    else:
        bbox_edges = None
        
    for i in trange(graph.nedge):
        #edge = graph.get_edge(i)
        npoints = nedgepoints[i]
        if npoints>=2:
            i0 = np.sum(nedgepoints[:i])
            i1 = i0+nedgepoints[i]
            epc = edgepoints[i0:i1]
            points_edge = edgepoints[i0:i1,0:2] #edge.coordinates[:,0:2]
            
            n0,n1 = nodecoords[conns[i]]
            if not np.all(n0==epc[0]) or not np.all(n1==epc[-1]):
                #breakpoint()
                print('Warning, node coordinate do not match edge coordinates!')
            
            rad = radii[i0:i1] # edge.scalars[edge.scalarNames.index(radName)]
            
            # Exclude capillaries, if required
            invalid = False
            if vtype is not None:
                if np.any(vtype[i0:i1]==2) and exclude_capillaries:
                    invalid = True
            
            inside = inside_fov(points_edge,fov)
            if np.any(inside) and not invalid:
                points = coord_to_index(points_edge,fov,dims)
                ins = False
                for j,pnt in enumerate(points):
                    if j>0:
                        if not np.all([inside[j],inside[j-1]]) and np.any([inside[j],inside[j-1]]) and bbox_edges is not None:
                            for bb in bbox_edges:
                                ins,int_point = geometry.segments_intersect_2d(points_edge[j-1],points_edge[j],bb[0],bb[1])
                                if ins:
                                    break
                            if ins:
                                if not inside[j]:
                                    points[j] = coord_to_index(int_point,fov,dims)
                                elif not inside[j-1]:
                                    points[j-1] = coord_to_index(int_point,fov,dims)
                    
                        if (inside[j] and inside[j-1]) or ins:
                            # Define radius
                            r = int(np.ceil(rad[j]))
                            
                            if radius_scale!=1.:
                                r *= radius_scale
                            
                            rpix_fl = r*2 / pixsize[0]
                            rpix = int(np.floor(rpix_fl))
                            rpix = np.clip(rpix,1,None)
                            
                            if fixed_radius is not None:
                                rpix = fixed_radius
                            
                            if rpix>0:
                                # Draw lines - value defined by height of vessel
                                col = 255 #int(points[j][2])
                                if rpix_fl<1.:
                                    col = int(255 * rpix_fl)
                                if vtype is None or vtype[i0]==0 or separate_artery_vein==False:
                                    enface_seg_image_draw_art.line((points[j-1][0],points[j-1][1],points[j][0],points[j][1]), fill=col, width=rpix)
                                elif vtype[i0]==1:
                                    enface_seg_image_draw_vein.line((points[j-1][0],points[j-1][1],points[j][0],points[j][1]), fill=col, width=rpix)
                                if False: #add_outline: # Not working!
                                    enface_seg_image_draw_art.line((points[j-1][0],points[j-1][1],points[j][0],points[j][1]), fill=0, width=int(np.ceil(rpix*1.1)))
                
    #breakpoint()
    if separate_artery_vein==False or vtype is None:
        image = np.asarray(enface_seg_image_art) #,dtype='int8')
    else:
        image = np.zeros([xdim,ydim,3])
        image[:,:,0] = np.asarray(enface_seg_image_art)
        image[:,:,2] = np.asarray(enface_seg_image_vein)
    #enface_seg_image = np.fliplr(np.rot90(enface_seg_image,3))
    
    if overlay_features and eye is not None:
        enface_seg_image = Image.new('I', (xdim, ydim), 0)      
        enface_seg_image_draw = ImageDraw.Draw(enface_seg_image)
        
        # Macula
        x0 = coord_to_index(eye.macula_centre[:2]-np.sqrt(2)*eye.fovea_radius/2.,fov,dims)
        x1 = coord_to_index(eye.macula_centre[:2]+np.sqrt(2)*eye.fovea_radius/2.,fov,dims)
        enface_seg_image_draw.ellipse((x0[0],x0[1],x1[0],x1[1]), fill=255)
        
        # Optic disc
        x0 = coord_to_index(eye.optic_disc_centre[:2]-np.sqrt(2)*eye.optic_disc_radius/2.,fov,dims)
        x1 = coord_to_index(eye.optic_disc_centre[:2]+np.sqrt(2)*eye.optic_disc_radius/2.,fov,dims)
        enface_seg_image_draw.ellipse((x0[0],x0[1],x1[0],x1[1]), fill=255)
        
        image[:,:,1] = np.asarray(enface_seg_image)
    
    return image

def create_enface(path='',opath='',ofile='',graph=None,xdim=500,ydim=500,view_size=arr([6000.,6000.]),eye=None,plot=False,centre=None,figsize=[7,7],exclude_capillaries=False,add_outline=True):   

    if graph is None:
        graph = spatialgraph.SpatialGraph()
        gfile = join(path,'retina_cco.am')
        graph.read(gfile)

    # LOAD EYE GEOMETRY FILE   
    if eye is None:
        eye = Eye()
        e_file = join(path,'retina_geometry.p')
        eye.load(e_file)
    
    if centre is None:
        centre = eye.fovea_centre[0:2]
    fov = arr([centre-(view_size/2.),centre+(view_size/2.)]).T
    
    dims = arr([xdim,ydim])

    image = enface_from_graph(graph,dims=dims,fov=fov)
    
    if plot:
        fig = plt.figure(figsize=figsize)
        nim = 1
        ind = 1
        ax = fig.add_subplot(1,nim,ind)
        ax.imshow(image,cmap='gray')
        plt.show()
    
    # Save images
    im = Image.fromarray(image.astype('int8'))
    if ofile=='':
        ofile = join(opath,'enface_segmentation.png')
    im = im.convert('RGB')
    im.save(join(opath,ofile))
    print(f'Saved enface segmentation to to {ofile}')
    
if __name__=='__main__':
    create_enface(path='',opath='')
    
