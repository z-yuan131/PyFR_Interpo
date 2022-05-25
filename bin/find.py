from math import pi

newmeshx = np.linspace(-2*pi,2*pi,200)
newmeshz = np.linspace(-2*pi/3,2*pi/3,100)
meshx, meshz = np.meshgrid(newmeshx,newmeshz)
meshy = meshx/meshx*0.1

newmesh = np.dstack((np.dstack((meshx,meshy)),meshz)).reshape((meshx.shape[0]*meshx.shape[1],3),order='F')

#hidecls = subclass_where(hide_och_catch(), name='sort')
import time

t = time.process_time()
idlist = hide(newmesh).getID()
elapsed_time = time.process_time() - t
