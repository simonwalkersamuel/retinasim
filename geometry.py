import numpy as np
import sys
import os
import re
import random
import json
import shutil
from matplotlib import pyplot as plt
import scipy.ndimage
import time
import numba as nb
from numba import jit, double
from numba.types import bool_, int_, float32

arr = np.asarray
cos,sin = np.cos,np.sin
xaxis,yaxis,zaxis = arr([1.,0.,0.]), arr([0.,1.,0.]), arr([0.,0.,1.])

# Return true if line segments AB and CD intersect
#def segments_intersect_2d(A,B,C,D):
#    def ccw(A,B,C):
#        return (C[1]-A[1]) * (B[0]-A[0]) >= (B[1]-A[1]) * (C[0]-A[0])
#    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
def segments_intersect_2d(A,B,C,D):

    from shapely.geometry import Point, LineString

    try:
        p1 = Point(A[0],A[1])
        p2 = Point(B[0],B[1])
        p3 = Point(C[0],C[1])
        p4 = Point(D[0],D[1])
    except Exception as e:
        print(e)
        breakpoint()

    l1 = LineString([p1,p2])
    l2 = LineString([p3,p4])
    try:
        coll = l1.intersects(l2)
    except Exception as e:
        print(e)
        breakpoint()
    
    pnt = None
    if coll:
        pnt = l1.intersection(l2)
        pnt = pnt.coords.xy
        pnt = arr([pnt[0][0],pnt[1][0]])

    return coll, pnt

# Faster vector normalisation?
@nb.njit(fastmath=True)
def norm(l): #,axis=None):
    s = 0.
    for i in range(l.shape[0]):
        s += l[i]**2
    return np.sqrt(s)
    
@nb.njit(fastmath=True)
def vector_norm(a,b):
    d = a - b
    return d / norm(d)

@nb.njit(fastmath=True)
def cross(vec1, vec2):
    """ Calculate the cross product of two 3d vectors. """
    a1, a2, a3 = double(vec1[0]), double(vec1[1]), double(vec1[2])
    b1, b2, b3 = double(vec2[0]), double(vec2[1]), double(vec2[2])
    result = np.zeros(3)
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result

@nb.njit(fastmath=True)
def rotate_3d(a,v):
    if np.all(a==0.):
        return v
        
    # Fast rotate 3d
    Rx = np.identity(3)
    Ry = np.identity(3)
    Rz = np.identity(3)
        
    # + (or - for -a)
    if a[0]!=0.:
        #R = np.eye(3)
        Rx[0,0] = np.cos(a[0])
        Rx[0,1] = -np.sin(a[0])
        Rx[1,0] = np.sin(a[0])
        Rx[1,1] = np.cos(a[0])
        v = np.dot(Rx,v)
    # ^ (or & for -a)
    if a[1]!=0.:
        #R = np.eye(3)
        Ry[0,0] = np.cos(a[1])
        Ry[0,2] = np.sin(a[1])
        Ry[2,0] = -np.sin(a[1])
        Ry[2,2] = np.cos(a[1]) 
        v = np.dot(Ry,v)
    # < (or / for -a)
    if a[2]!=0.:
        #Rz = np.eye(3)
        Rz[1,1] = np.cos(a[2])
        Rz[1,2] = -np.sin(a[2])
        Rz[2,1] = np.sin(a[2])
        Rz[2,2] = np.cos(a[2])
        v = np.dot(Rz,v)
    return v

@nb.njit(fastmath=True)
def translate(l,coord,dir):
    M = np.eye(4)
    ch = np.append(coord,1.)
    v = l*dir/norm(dir)
    M[:3,3] = v
    rh = np.dot(M,ch)
    return rh[0:3]  

@nb.njit(fastmath=True)    
def align_vectors(vec1,vec2,homog=False,epsilon=1e-9):
   """
   Calculate rotation matrix to align vector a with b
   """
   d = 3
   if homog:
      d = 4
   if np.all(vec1==vec2):
       return np.eye(d)

   a, b = (vec1 / norm(vec1)).reshape(3), (vec2 / norm(vec2)).reshape(3)
   v = cross(a, b)
   c = np.dot(a, b)
   s = norm(v)
   if s==0:
       if c>=0.: # parallel
           return np.eye(d)
       else: #anti-parallel
           # If pointing along z-axis
           if vec2[2]!=0 and vec2[2]==-vec1[2]:
               res = np.eye(d)
               res[0,0],res[2,2] = -1,-1
               return res
           # Opposite directions along z
           elif np.dot(vec1,vec2)>-1-epsilon and np.dot(vec1,vec2)<-1+epsilon:
               res = np.eye(d)
               res[2,2] = -1.
               return res
           else:
               res = np.eye(d)
               res[0,0] = -(vec1[0]**2-vec1[1]**2)
               res[0,1] = -2*vec1[0]*vec1[1]
               res[1,0] = -2*vec1[0]*vec1[1]
               res[1,1] = vec1[0]**2-vec1[1]**2
               res[2,2] = -(1.-vec1[2]**2)
               return res
   kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
   rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
   if homog:
       res = np.eye(4)
       res[0:3,0:3] = rotation_matrix
       return res
   else:
       return rotation_matrix    
    
#@nb.njit(fastmath=True)    
def vector_angle(U,V):
    # Angle between two vectors (one-sided)
    return np.arccos(np.abs(np.dot(V, U)) / (np.linalg.norm(V) * np.linalg.norm(U)))
    
@nb.njit(fastmath=True) 
def rotate_about_axis(vector,axis,theta,R=None,centre=None):
    """ 
    Rotate a 3D vector by theta *deg* about an arbitrary axis, centred at origin
    """
    if R is None:
        # Rotate axis to z-axis
        R1 = np.eye(4)
        R1[0:3,0:3] = align_vectors(axis,arr([0.,0.,1.]))
        R1inv = np.linalg.inv(R1)
        # Rotate about z
        R2 = np.eye(4)
        a = np.deg2rad(theta)
        R2[0,0],R2[0,1],R2[1,0],R2[1,1] = cos(a),-sin(a),sin(a),cos(a)
        R = R1inv.dot(R2.dot(R1))

    # Homogeneous coordinates
    vh = np.append(vector,[1.])            
    
    if centre is not None:
        Tr = np.eye(4)
        Tinv = np.eye(4)
        Tr[:3,3] = -centre
        Tinv[:3,3] = centre
        return Tinv.dot(R.dot(Tr.dot(vh)))[0:3]
    else:
        return R.dot(vh)[0:3]
        
def rotate_points_about_axis(coords,axis,theta,R=None,centre=arr([0.,0.,0.])):
    """ 
    Rotate 3D coordinates by theta *deg* about an arbitrary axis, centred at origin
    """
    if R is None:     
        # Transform to rotate supplied axis to z-axis
        R1 = np.eye(4)
        axis = axis / norm(axis)
        if np.all(axis==zaxis):
            R1inv = np.eye(4)
        else:
            R1[0:3,0:3] = align_vectors(axis,zaxis)
            R1inv = np.linalg.inv(R1)
        # Rotation about z
        R2 = np.eye(4)
        a = np.deg2rad(theta)
        R2[0,0],R2[0,1],R2[1,0],R2[1,1] = cos(a),-sin(a),sin(a),cos(a)
        # Rotation from z-axis back to supplied axis
        R = R1inv.dot(R2.dot(R1))

    # Homogeneous coordinates
    coords_h = np.zeros([coords.shape[0],4],dtype=coords.dtype)
    coords_h[:,0:3] = coords
    
    if centre is not None:
        # Translate points to origin
        Tr = np.eye(4)
        Tr[:3,3] = -centre
        # Return points to original frame after rotation
        Tinv = np.eye(4)
        Tinv[:3,3] = centre
        for i,coord in enumerate(coords_h):
            #coords_h[i,:] = Tinv.dot(R.dot(Tr.dot(coord)))
            coord[0:3] -= centre
            coord = np.dot(R,coord)
            coord[0:3] += centre
            coords_h[i,:] = coord
        return coords_h[:,0:3]
    else:
        for i,coord in enumerate(coords_h):
            coords_h[i,:] = R.dot(coord)
        return coords_h[:,0:3]


#@nb.njit(fastmath=True) 
def rotate_about_axiswith_matrix(vector,axis,R=None,centre=None):
    """ 
    Rotate a 3D vector by theta *deg* about an arbitrary axis, centred at origin
    """

    # Homogeneous coordinates
    vh = np.append(vector,[1.])
    
    #if centre is not None:
    Tr = np.eye(4)
    Tinv = np.eye(4)
    Tr[:3,3] = -centre
    Tinv[:3,3] = centre
    return Tinv.dot(R.dot(Tr.dot(vh)))[0:3]
    
@nb.njit(fastmath=True)       
def rotate_3d_array_matrix(R,centre=[0.,0.,0.]):

    tr = np.eye(4)
    tr[0:3,3] = [-x for x in centre]
    tri = np.linalg.inv(tr) 
    
    Rh = np.eye(4)
    Rh[0:3,0:3] = R
    Rhi = np.linalg.inv(Rh)
    
    return tr*Rhi*tri 
   
@nb.njit(fastmath=True)
def line_sphere_collision(self,centre,radius,x0,x1):
    """The ray t value of the first intersection point of the
    ray with self, or None if no intersection occurs"""
    
    length = norm(x1-x0)
    dir = (x1-x0) / length
    
    q = centre - x0
    vDotQ = np.dot((x1-x0), q)
    squareDiffs = np.dot(q, q) - radius*radius
    discrim = vDotQ * vDotQ - squareDiffs
    
    if discrim >= 0:
        root = np.sqrt(discrim)
        t0 = (vDotQ - root)
        t1 = (vDotQ + root)

        x2 = x0+(dir*t0)
        x3 = x0+(dir*t1)
        t0onseg = (t0>=0 and t0<=length) # norm(x2-x0)<=norm(x1-x0) and np.dot(x1-x0,x2-x0)>=0.
        t1onseg = (t1>=0 and t1<=length) # norm(x3-x0)<=norm(x1-x0) and np.dot(x1-x0,x3-x0)>=0.
        #print(x2,x3)

        if t0onseg and t1onseg:
            if t0 < t1:
                return True, [x2, x3]
            else:
                return True, [x3, x2]
        elif t0onseg:
            return True, [x2]
        elif t1onseg:
            return True, [x3]
    return False, []     

@nb.njit(fastmath=True)
def inside_cone(self, p, a, b, theta)        :
    lp_dist = self.point_segment_distance(p,a,b)
    lp_len = np.dot(b-a,p-a) # length of nearest point along l0-l1
    lp1 = a + (b-a)*lp_len
    cone_rad = np.tan(theta) * lp_len
    if lp_dist<cone_rad:
        return True
    return False 
    
#def point_inside_cylinder(A,B,r,p):
#
#    l = norm(B - A)
#    AB = (B - A) / l

#@nb.njit(fastmath=True)
#def point_in_cylinder(pt1, pt2, r, q):
#    vec = pt2 - pt1
#    const = r * norm(vec)
#    return np.dot(q - pt1, vec) >= 0 and np.dot(q - pt2, vec) <= 0 and norm(np.cross(q - pt1, vec)) <= const
    
#@nb.njit(fastmath=True)
def point_in_cylinder(pt1, pt2, r, q):
    """
    pt1: start point(s) of cylinder(s)
    pt2: end point(s) of cylinder(s)
    r: radius of cylinder(s)
    q: point coordinates
    """
    if np.all(pt1==pt2) or np.all(r<=0.):
        return False
    vec = pt2 - pt1
    const = r * np.linalg.norm(vec)
    e1 = np.dot(q - pt1, vec.transpose()) >= 0
    e2 = np.dot(q - pt2, vec.transpose()) <= 0
    e3 = np.linalg.norm(np.cross(q - pt1, vec),axis=1) <= const
    return e1 & e2 & e3

#@nb.njit(fastmath=True)    
def point_segment_distance(p, a, b):

    """Distance between a point (p) and a line segment (a,b)
    """

    # normalized tangent vector
    d = np.divide(b - a, norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = cross(p - a, d)

    return np.hypot(h, norm(c))

#@nb.njit(fastmath=True)
def segment_intersection(a,b,c,d):   

    # a: segment 1 start coordinate (x,y,z)
    # b: segment 1 end coordinate (x,y,z)
    # c: segment 2 start coordinate (x,y,z)
    # d: segment 2 end coordinate (x,y,z)

    A = b[0]-a[0]
    B = c[0]-d[0]
    C = c[0]-a[0]
    D = b[1]-a[1]
    E = c[1]-d[1]
    F = c[1]-a[1]

    #find t and s using formula
    t = (C*E-F*B)/(E*A-B*D)
    s = (D*C-A*F)/(D*B-A*E)

    #check if third equation is also satisfied(we have 3 equations and 2 variable
    if ((t*(b[2]-a[2])+s*(c[2]-d[2]))==c[2]-a[2]):
        if(0<=t<=1 and 0<=s<=1):
           #print('line segments intersect')
           return True
        else:
           #print ('line segments intersect on being produced')
           return True
    else:
        return False
        
#@nb.njit(fastmath=True)
def segment_intersection_multi(a,b,c,d,epsilon=1e-12):   

    # a: segment 1 start coordinate (x,y,z)
    # b: segment 1 end coordinate (x,y,z)
    # c: n segment start coordinates (n,x,y,z)
    # d: n segment end coordinates (n,x,y,z)

    A = b[0]-a[0]
    B = c[0]-d[0]
    C = c[0]-a[0]
    D = b[1]-a[1]
    E = c[1]-d[1]
    F = c[1]-a[1]

    #find t and s using formula
    t = (C*E-F*B)/((E*A-B*D)+epsilon)
    s = (D*C-A*F)/((D*B-A*E)+epsilon)

    #check if third equation is also satisfied(we have 3 equations and 2 variable
    res = np.zeros(t.shape[0],dtype='bool')
    #res = np.zeros_like(t, bool_)
    res[(t>=0) & (t<=1) & (s>=0) & (s<=1) & ((t*(b[2]-a[2])+s*(c[2]-d[2]))==c[2]-a[2])] = True
      
    return res

@nb.njit(fastmath=True)    
def point_line_closest(p, a, b, on_line=False,snap_to_vert=False):

    # On a line segment defined by points a and b, returns the closest point from p
    # on_line: flag to specify whether the point must be on the line, or if to snap to a or b if outside the segment

    ab = b - a
    ab /= norm(ab)
    
    ap = p - a
    d = np.dot(ap, ab)

    pnt = a + ab * d
    
    # Check if point is outside line segment
    if np.dot(b-a,pnt-b)>0.:
        if on_line:
            return None
        if snap_to_vert:
            pnt = b
    if np.dot(b-a,a-pnt)>0.:
        if on_line:
            return None
        if snap_to_vert:
            pnt = a

    return pnt

#@nb.njit(fastmath=True)    
def line_plane_intersection(pp,pn,lp,ld,epsilon=1e-6):
    """
    Test where a line segment passes through a plane
    pp: point on the plane
    pn: normal vector of the plane
    lp: point on the line
    ld: Direction vector of line segment
    """
    ndotu = np.dot(pn,ld)
    if abs(ndotu)<epsilon: # Parallel
        return False,None
    w = lp - pp
    si = -np.dot(pn,w) / ndotu
    return True,lp+(ld*si) #w + si * ld + pp

#@nb.njit(fastmath=True)    
def line_circle_intersection(x1,x2,Q,Qn,r):
    """
    Test where a line passes through end caps
    Q: circle centre coordinate
    Qn: circle normal
    r: circle radius
    x1,x2: line start/end coords
    """
    lInt,lpi = line_plane_intersection(Q,Qn,x1,x2-x1)
    if lInt:
        return norm(lpi-Q)<=r,lpi
    else:
        return False,None

#@nb.njit(fastmath=True)        
#def point_in_cylinder(pt1, pt2, r, q):
#    """
#    Test whether point q lies in a cylinder defined by
#    end points pt1 and pt2 and radius r
#    """
#    vec = pt2 - pt1
#    const = r * norm(vec)
#    return np.dot(q - pt1, vec) >= 0 and np.dot(q - pt2, vec) <= 0 and norm(cross(q - pt1, vec)) <= const
    
def point_on_segment_2d(p, q, r):

    # Given three colinear points p, q, r, the function checks if  
    # point q lies on line segment 'pr'   
    if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
           (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))): 
        return True
    return False
  
def orientation(p, q, r): 
    # to find the orientation of an ordered triplet (p,q,r) 
    # function returns the following values: 
    # 0 : Colinear points 
    # 1 : Clockwise points 
    # 2 : Counterclockwise 
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/  
    # for details of below formula.  
      
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1])) 
    if (val > 0): 
        # Clockwise orientation 
        return 1
    elif (val < 0): 
        # Counterclockwise orientation 
        return 2
    else: 
        # Colinear orientation 
        return 0
  
# The main function that returns true if  
# the line segment 'p1q1' and 'p2q2' intersect. 
def segment_collision_2d(p1,q1,p2,q2,verbose=False): 
      
    # Find the 4 orientations required for  
    # the general and special cases 
    o1 = orientation(p1, q1, p2) 
    o2 = orientation(p1, q1, q2) 
    o3 = orientation(p2, q2, p1) 
    o4 = orientation(p2, q2, q1) 
  
    # General case 
    if ((o1 != o2) and (o3 != o4)):
        if verbose:
            print('General case ({},{},{},{})'.format(o1,o2,o3,o4)) 
        return True
  
    # Special Cases 
  
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1 
    if ((o1 == 0) and point_on_segment_2d(p1, p2, q1)): 
        if verbose:
            print('p1 , q1 and p2 are colinear and p2 lies on segment p1q1')
        return True
  
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1 
    if ((o2 == 0) and point_on_segment_2d(p1, q2, q1)): 
        if verbose:
            print('p1 , q1 and q2 are colinear and q2 lies on segment p1q1 ')
        return True
  
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2 
    if ((o3 == 0) and point_on_segment_2d(p2, p1, q2)): 
        if verbose:
            print('p2 , q2 and p1 are colinear and p1 lies on segment p2q2 ')
        return True
  
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2 
    if ((o4 == 0) and point_on_segment_2d(p2, q1, q2)): 
        if verbose:
            print('p2 , q2 and q1 are colinear and q1 lies on segment p2q2 ')
        return True
  
    # If none of the cases 
    return False

#@nb.njit(fastmath=True)    
def line_cylinder_collision(A,B,r,start_coord,end_coord,optimize=True,verbose=False):
    
    """ Detect a collision between a line and a cylinder
    A = start coordinate of central vertex of cylinder
    B = end coordinate of central vertex of cylinder
    r = radius of cylinder
    start_coord = start coordinate of line
    end_coord = end coordinate of line
    
    Returns:
    cap_intersection_coords: Coordinates of collisions between end cap circles (if any - both None otherwise). Ordered start/end
    body_intersection_coords: Coordinates of collisions between line and cylinder barrel (if any). Not ordered.
    """
    dir = end_coord - start_coord
    
    cap_intersections = [False,False]
    cap_intersection_coords = [None,None]
    body_intersections = [False,False]
    body_intersection_coords = []

    # Check end circles
    Qn = (A - B)/norm(A-B)
    lc1,lp1 = line_circle_intersection(start_coord,end_coord,A,Qn,r)
    lc2,lp2 = line_circle_intersection(start_coord,end_coord,B,Qn,r)
    cap_intersections = [lc1,lc2]
    cap_intersection_coords = [lp1,lp2]
    
    # Line intersects both end circles
    if lc1 and lc2:
        intersection = np.zeros((2,3))
        intersection[0,:] = lp1
        intersection[1,:] = lp2
        return cap_intersection_coords, body_intersection_coords

    # Solution : http://www.gamedev.net/community/forums/topic.asp?topic_id=467789

    # Find intersection points on cylinder walls
    AB = B - A
    AO = start_coord - A
    AOxAB = cross(AO,AB)
    VxAB  = cross(dir,AB)
    ab2 = np.dot(AB,AB)
    a = np.dot(VxAB,VxAB)
    b = 2 * np.dot(VxAB,AOxAB)
    c = np.dot(AOxAB,AOxAB) - (r*r * ab2)
    d = b * b - 4 * a * c
    if (d <= 0):
        # No real roots
        if verbose:
            print('No real roots')
        return cap_intersection_coords, body_intersection_coords

    t0,t1 = (-b - np.sqrt(d)) / (2 * a),(-b + np.sqrt(d)) / (2 * a)
    if t0<0. and t1<0.:
        if verbose:
            print('No real roots')
        return cap_intersection_coords, body_intersection_coords

    if t0>=0.:
        intersection0 = start_coord + dir * t0 # intersection point
        projection0 = A + (np.dot(AB,(intersection0 - A)) / ab2) * AB # intersection projected onto cylinder axis
        test0_1 = (norm(projection0 - A) + norm(B - projection0)) <= norm(AB)
        test0_2 = norm(dir)>=norm(intersection0-start_coord)
        test_0 = test0_1 and test0_2
        normal0 = (intersection0 - projection0)
        nrm = norm(normal0)
        if nrm>0:
            normal0 /= norm(normal0)
    else:
        test_0 = False
        
    if t1>=0.:
        intersection1 = start_coord + dir * t1 # intersection point
        projection1 = A + (np.dot(AB,(intersection1 - A)) / ab2) * AB # intersection projected onto cylinder axis
        test1_1 = (norm(projection1 - A) + norm(B - projection1)) <= norm(AB)
        test1_2 = norm(dir)>=norm(intersection1-start_coord)
        test_1 = test1_1 and test1_2
        normal1 = (intersection1 - projection1)
        nrm = norm(normal1)
        if nrm>0.:
            normal1 /= nrm
    else:
        test_1 = False

    if test_0 and test_1:
        body_intersection_coords = [intersection0,intersection1]
        if verbose:
            print('Two body intersections')
    elif test_0 and not test_1:
        body_intersection_coords = [intersection0]
        if verbose:
            print('One body intersection (1st)')
    elif not test_0 and test_1:
        body_intersection_coords = [intersection1]
        if verbose:
            print('One body intersection (2nd)')            
    elif not test_0 and not test_1:
        body_intersection_coords = []
        if verbose:
            print('No intersections')
    
    return cap_intersection_coords, body_intersection_coords

#@nb.njit(fastmath=True)        
def segment_cylinder_collision(A,B,r,start_coord,end_coord,line_dir=None,optimize=True):
    
    # Do not use provided line direction - recalculate
    cap_intersection_coords, body_intersection_coords = line_cylinder_collision(A,B,r,start_coord,end_coord,optimize=True)
    
    intersections,intersection_types = [],[]
    if cap_intersection_coords[0] is not None and norm(cap_intersection_coords[0]-A)<+r:
        intersections.append(cap_intersection_coords[0])
        intersection_types.append(1)
    if cap_intersection_coords[1] is not None and norm(cap_intersection_coords[1]-B)<+r:
        intersections.append(cap_intersection_coords[1])
        intersection_types.append(1)
 
    if len(body_intersection_coords)>0:
        for intr in body_intersection_coords:
            intersections.append(intr)
            intersection_types.append(2)
        
    intersections = np.asarray(intersections)
    intersection_types = np.asarray(intersection_types)
    
    inside = [point_in_cylinder(A, B, r, start_coord), point_in_cylinder(A, B, r, end_coord)]
    
    if np.all(inside) or np.all(intersections==None):
        return [], [], inside

    flag = np.zeros(len(intersections))
    for i,intersection in enumerate(intersections):
        if intersection is not None:
            x0_0 = np.asarray(end_coord-start_coord,dtype=np.float64).squeeze()
            x1_0 = np.asarray(intersection-start_coord,dtype=np.float64).squeeze()
            if np.dot(x0_0,x1_0)>=0. and norm(x0_0)>=norm(x1_0):
                flag[i] = 1

    return intersections[flag==1], intersection_types[flag==1], inside
    
def cylinder_collision(A1,B1,r1,A2,B2,r2,n=16):

    #vector in direction of axis
    v2 = A2 - B2
    mag = norm(v2)
    v2 /= mag

    #make some vector not in the same direction as v2
    not_v2 = np.array([1, 0, 0])
    if np.abs(np.dot(v2,not_v2))==1.: #(v2 == not_v2).all():
        not_v2 = np.array([0, 1, 0])

    #make vector perpendicular to v2
    n1 = cross(v2, not_v2)
    n1 /= norm(n1)
    #make unit vector perpendicular to v2 and n1
    n2 = np.cross(v2, n1)
    n2 /= norm(n2)
    
    intersection_store, intersection_type_store, inside_store = [],[],[]
    
    probe_store = []
    
    # Centre line collisions
    probe_store.append([A2,B2])
    intersections, intersection_types, inside = segment_cylinder_collision(A1,B1,r1,A2,B2)
    if len(intersections)>0 or np.any(inside):
        return True,probe_store,intersections
    
    # Test for collisions away from centre line (brute force)
    # Set n=0 to skip this step
    for t in np.linspace(0, 2 * np.pi, n):
        A2p = np.asarray([A2[i] + r2 * np.sin(t) * n1[i] + r2 * np.cos(t) * n2[i] for i in [0,1,2]])
        B2p = np.asarray([B2[i] + r2 * np.sin(t) * n1[i] + r2 * np.cos(t) * n2[i] for i in [0,1,2]])
        probe_store.append([A2p,B2p])
        intersections, intersection_types, inside = segment_cylinder_collision(A1,B1,r1,A2p,B2p)
        if len(intersections)>0:# or np.any(inside):
            return True,probe_store,intersections
    return False,probe_store,[]
