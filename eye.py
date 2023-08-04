import numpy as np
arr = np.asarray
import dill

# Default values
ext = 35000./2.
xoffset = 0
DOMAIN = arr([[-ext+xoffset,ext+xoffset],[-ext,ext],[-ext,ext]]).astype('float')

uni = np.random.uniform

class Eye(object):

    def __init__(self, **kwargs):

        self.eye = 'right'
        
        self.optic_disc_centre = arr([0.,0.,0.])
        self.optic_cup_diameter = uni(0.7,1.2)*1000. # um
        self.optic_disc_radius = self.optic_cup_diameter*uni(1.1,1.5) #1800. #um
        self.occular_radius = 24000. #np.random.normal(24000.,1000.) # um
        self.occular_centre = arr([0.,0.,self.occular_radius])
        
        # Distance between optic nerve and fundus
        self.dfd = uni(3500.,5500.)
        self.fovea_centre = self.optic_disc_centre.copy()
        self.fovea_centre[0] += self.dfd

        self.fovea_radius = 500. + np.random.normal(0.,20.) #350./2. # um https://en.wikipedia.org/wiki/Fovea_centralis#:~:text=The%20fovea%20is%20a%20depression,area%20without%20any%20blood%20vessels).
        self.macula_centre = self.fovea_centre
        self.macula_radius = 2500. + np.random.normal(0.,200.)
        
        # L-system variables
        self.T_branching_factor = 20. #0.2
        self.Y_branching_factor = 30.
        self.segment_fraction = 1.

        # Retinal artery diameter 
        self.adiameter = 135. + np.random.normal(0.,15.64) # Retinal artery
        self.amndiameter = 100.
        # Retinal vein diameter 
        self.vdiameter = 151. + np.random.normal(0.,15.22) # Retinal vein
        
        self.domain = DOMAIN
        
        for k,v in zip(kwargs.keys(),kwargs.values()):
            setattr(self,k,v)
            
    def save(self, filename):
        with open(filename, 'wb') as f:
            dill.dump(self.__dict__,f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = dill.load(f)
        
        for k,v in zip(data.keys(),data.values()):
            try:
                setattr(self,k,v)
            except Exception as e:
                print(e)
