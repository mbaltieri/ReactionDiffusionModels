#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:36:34 2018

Grey-Scott model found online, details below
Munafo's fixed spot (negaton?)

@author: manuelbaltieri
"""

# Reaction-Diffusion Simulation Using Gray-Scott Model
# https://en.wikipedia.org/wiki/Reaction-diffusion_system
# http://www.labri.fr/perso/nrougier/teaching/numpy/numpy.html#
# FB - 20160130
#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Reaction Diffusion : Gray-Scott model

References:
----------
Complex Patterns in a Simple System
John E. Pearson, Science 261, 5118, 189-192, 1993.

Encode movie
------------

ffmpeg -r 30 -i "tmp-%03d.png" -c:v libx264 -crf 23 -pix_fmt yuv420p bacteria.mp4
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import animation

#from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from jpype import *


plt.close('all')

# check if JVM was started
def init_jvm(jvmpath=None):
    if isJVMStarted():
        return
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

runs = 1
T = 2000
dt = 1
iterations = int(T/dt)


# Parameters from http://www.aliensaint.com/uo/java/rd/
# -----------------------------------------------------
n  = 50
#Du, Dv, F, k = 0.16, 0.08, 0.035, 0.065 # Bacteria 1
#Du, Dv, F, k = 0.14, 0.06, 0.035, 0.065 # Bacteria 2
#Du, Dv, F, k = 0.16, 0.08, 0.060, 0.062 # Coral
#Du, Dv, F, k = 0.19, 0.05, 0.060, 0.062 # Fingerprint
#Du, Dv, F, k = 0.10, 0.10, 0.018, 0.050 # Spirals
#Du, Dv, F, k = 0.12, 0.08, 0.020, 0.050 # Spirals Dense
#Du, Dv, F, k = 0.10, 0.16, 0.020, 0.050 # Spirals Fast
#Du, Dv, F, k = 0.16, 0.08, 0.020, 0.055 # Unstable
#Du, Dv, F, k = 0.16, 0.08, 0.050, 0.065 # Worms 1
#Du, Dv, F, k = 0.16, 0.08, 0.054, 0.063 # Worms 2
#Du, Dv, F, k = 0.16, 0.08, 0.035, 0.060 # Zebrafish
#Du, Dv, F, k = 0.2, 0.1, 0.06, 0.06093 #
Du, Dv, F, k = 0.164, 0.082, 0.062, 0.06093 #



Z = np.zeros((n+2,n+2), [('U', np.double), ('V', np.double)])
U,V = Z['U'], Z['V']
u,v = U[1:-1,1:-1], V[1:-1,1:-1]

r = 4
r_2 = int(r/2)
square_side = 40
square_side_2 = int(square_side/2)
#u[...] = 1.0
u[...] = .5
v[...] = .3
m = int(n/2)
#U[m-r:m+r,m-r:m+r] = 0.50
#V[m-r:m+r,m-r:m+r] = 0.25
#V[m-r-r_2:m-r_2-2,m-r:m+r] = 0.0
#V[m-r_2:m+r_2,m:m+r] = 0.0
#V[m+r_2+2:m+r+r_2,m-r:m+r] = 0.0

# U shape
#V[m-r:m-r_2,m-r:m+r] = 0.0
#V[m-r_2:m+r_2,m:m+r] = 0.0
#V[m:m+r_2,m-r:m+r] = 0.0

# Spot
#V[m-r:m-r_2,m-r:m+r] = 0.0
V[m-r_2:m+r_2,m-r_2:m+r_2] = 0.0
#V[m:m+r_2,m-r:m+r] = 0.0

u += 0.005*np.random.random((n,n))
v += 0.005*np.random.random((n,n))

#for i in range(20):
#    n1 = np.random.randint(n-2*r)
#    n2 = np.random.randint(n-2*r)
#    n3 = np.random.randint(n-2*r)
#    n4 = np.random.randint(n-2*r)
#    
#    U[n1-r:n1+r,n2-r:n2+r] = np.random.rand()
#    V[n3-r:n3+r,n4-r:n4+r] = np.random.rand()

#for i in range(5):
#    m = np.random.randint(n-2*r)
#    
#    V[m-r:m-r_2,m-r:m+r] = 0.0
#    V[m-r_2:m+r_2,m:m+r] = 0.0
#    V[m:m+r_2+6,m-r:m+r] = 0.0



plt.ion()

size = np.array(Z.shape)
dpi = 36.0
figsize= 8*size[1]/float(dpi),8*size[0]/float(dpi)
#fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, dpi=dpi, facecolor="white")
fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
plt.xticks([]), plt.yticks([])
plt.subplot(111)
im = plt.imshow(V, interpolation='bicubic', cmap=plt.cm.gray_r)
im.set_clim(vmin=0, vmax=.45)
#plt.xticks([]), plt.yticks([])
#plt.subplot(312)
#im2 = plt.imshow(U, interpolation='bicubic', cmap=plt.cm.gray_r)
##im2.set_clim(vmin=0, vmax=.45)
#plt.subplot(313)
#im3 = plt.imshow(U, interpolation='bicubic', cmap=plt.cm.gray_r)
#im2.set_clim(vmin=0, vmax=.45)
#
# save entire history
v_hist = np.zeros((runs,iterations,n,n))
v_binary_hist = np.zeros((runs,iterations,n,n))
membrane_binary_hist = np.zeros((runs,iterations,n,n))
membrane_hist = np.zeros((runs,iterations,n,n))

u_hist = np.zeros((runs,iterations,n,n))

frame_spot_histV = np.zeros((runs,iterations,square_side,square_side))
frame_spot_histU = np.zeros((runs,iterations,square_side,square_side))

centroid = np.array([0,0])

u_rest = 1.0

for j in range(runs):
    print(j)
    for i in range(iterations):
#        if i == 5000:
##            u_rest = 1.0005
#            F = .057
        print(i)
        Lu = (                 U[0:-2,1:-1] +
              U[1:-1,0:-2] - 4*U[1:-1,1:-1] + U[1:-1,2:] +
                               U[2:  ,1:-1] )
        Lv = (                 V[0:-2,1:-1] +
              V[1:-1,0:-2] - 4*V[1:-1,1:-1] + V[1:-1,2:] +
                               V[2:  ,1:-1] )
    
        uvv = u*v*v
        du = Du*Lu - uvv +  F   *(u_rest-u)
        u += (du)
        dv = Dv*Lv + uvv - (F+k)*v    
        v += (dv)
        
        # find spot
#        binary_v = np.where(v<.15, 1, 0)
        binary_v = np.where(v<.20, 1, 0)
        binary_v[:10,:] = 0
        binary_v[-10:,:] = 0
        binary_v[:,:10] = 0
        binary_v[:,-10:] = 0
#        
#        # find centroid of spot
        coord = np.argwhere(binary_v>.5)
        length = coord.shape[0]
        centroid[0] = np.sum(coord[:,0])/length
        centroid[1] = np.sum(coord[:,1])/length
#        
#        # find membrane
#        binary_membrane = np.where((v>=.15) & (v<.30), 1, 0)
#        binary_membrane[:centroid[0]-square_side_2,:] = 0
#        binary_membrane[centroid[0]+square_side_2:,:] = 0
#        binary_membrane[:,:centroid[1]-square_side_2] = 0
#        binary_membrane[:,centroid[1]+square_side_2:] = 0
        
        # save frame around spot
        frame_spot_histV[j,i,:,:] = v[centroid[0]-square_side_2:centroid[0]+square_side_2,centroid[1]-square_side_2:centroid[1]+square_side_2]
        frame_spot_histU[j,i,:,:] = u[centroid[0]-square_side_2:centroid[0]+square_side_2,centroid[1]-square_side_2:centroid[1]+square_side_2]
        
        # save spot and membrane
        v_hist[j,i,:,:] = v
        u_hist[j,i,:,:] = u
#        v_binary_hist[j,i,:,:] = binary_v
#        membrane_binary_hist[j,i,:,:] = binary_membrane
#        membrane_hist[j,i,:,:] = np.zeros((runs,iterations,n,n))
        
        

        if i % 100 == 0:
            im.set_data(V)
        #        im.set_clim(vmin=V.min(), vmax=V.max())
            
#            im2.set_data(U)
#        #        im2.set_clim(vmin=U.min(), vmax=U.max())
##        
#            im3.set_data(dv)
#            im3.set_clim(vmin=dv.min(), vmax=dv.max())
##            
            plt.draw()
            plt.pause(.01)
#
plt.ioff()
plt.show()

print("Simulation - done!")









# predictive information + active information storage + transfer entropy analysis

# Change location of jar to match yours:
jarLocation = "../../infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
#startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)
init_jvm()

# details of information being analysed
starting_point_time_series = 0
size = n


## Create a PI calculator and run it:
#piCalcClass = JPackage("infodynamics.measures.continuous.kraskov").PredictiveInfoCalculatorKraskov
#piCalc = piCalcClass()
#piCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
#
#piCalc.setProperty("k", "4") # Use Kraskov parameter K=4 for 4 nearest points
#
## chemical V
#piAverageV = np.zeros((runs,size,size))
#piLocalV = np.zeros((runs,iterations-starting_point_time_series,size,size))
#
## chemical U
#piAverageU = np.zeros((runs,size,size))
#piLocalU = np.zeros((runs,iterations-starting_point_time_series,size,size))
#
#
#for k in range(runs):
##    print(k)
#    for i in range(size):
#        for j in range(size):
#            print(k, ' - ', i, ' - ', j)
#            
#            # PI calculator - V chemical
#            sourceArray = v_hist[0,:,i,j].tolist()            # entire grid
##            sourceArray = frame_spot_histV[0,starting_point_time_series:,i,j].tolist()    # fixed frame around moving agent
#            piCalc.initialise(10) # Use history length 1 (Schreiber k=1)
#            piCalc.setObservations(JArray(JDouble, 1)(sourceArray))
#            piLocalV[k,:,i,j] = piCalc.computeLocalOfPreviousObservations()
#            piAverageV[k,i,j] = piCalc.computeAverageLocalOfObservations()
#            
#            # PI calculator - U chemical
##            sourceArray = u_hist[0,starting_point_time_series:,i,j].tolist()            # entire grid
###            sourceArray = frame_spot_histU[0,starting_point_time_series:,i,j].tolist()    # fixed frame around moving agent
##            piCalc.initialise(1) # Use history length 1 (Schreiber k=1)
##            piCalc.setObservations(JArray(JDouble, 1)(sourceArray))
###            piLocalU[k,i,j,:] = piCalc.computeLocalOfPreviousObservations()
##            piAverageU[k,i,j] = piCalc.computeAverageLocalOfObservations()
#
## chemical V
#piAverageV_avg_runs = np.average(piAverageV, axis=0)
#piLocalV_avg_runs = np.average(piLocalV, axis=0)
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
## Plot the surface
#x = range(size)
#y = range(size)
#x, y = np.meshgrid(x, y)
##ax = plt.contourf(x, y, piAverageV_avg_runs, 50, cmap=cm.jet)           # average
#surf = ax.plot_surface(x, y, piAverageV_avg_runs, cmap=cm.jet, linewidth=0)
#plt.title('Average Predictive Information')
##cbar = fig.colorbar(ax, orientation='vertical')
##cbar.set_clim(2.0, 7.0)
#
#
#plt.ion()
#fig2 = plt.figure()
#for i in range(iterations):
#    if i % 100 == 0:
#        print(i)
#        plt.clf()
##        plt.contourf(x, y, piLocalV_avg_runs[i,:,:], 25, cmap=cm.jet)
#        pi = plt.imshow(piLocalV_avg_runs[i,:,:], interpolation='bicubic', cmap=cm.jet)
#        pi.set_clim(vmin = np.min(piLocalV_avg_runs[i,:,:]), vmax = np.max(piLocalV_avg_runs[i,:,:]))
##        cbar = plt.colorbar();
##        cbar.set_clim(vmin = np.min(piLocalV_avg_runs[i,:,:]), vmax = np.max(piLocalV_avg_runs[i,:,:]))
##        plt.title('Predictive Information, i = ', i)
#        plt.show()
#        plt.pause(.01)
#
#plt.ioff()
#
#
##def animate(i): 
##    z = piLocalV_avg_runs[i,:,:]
##    cont = plt.contourf(x, y, z, 25)
##    
###    if (tslice == 0):
###        plt.title(r't = %1.2e' % t[i] )
###    else:
###        plt.title(r't = %i' % i)
###    plt.title('Predictive Information, i = ', i)
##    return cont  
##
##fig = plt.figure()
##
##plt.title('Predictive Information')
##anim = animation.FuncAnimation(fig, animate, interval=10)
##fig.colorbar(animate, orientation='vertical')
#
#
### chemical U
##piAverageU_avg = np.average(piAverageU, axis=0)
##fig2 = plt.figure()
##ax2 = fig2.add_subplot(111, projection='3d')
##
### Plot the surface
##ax2.plot_surface(x, y, piAverageU_avg, cmap=cm.jet)
##
##
##for i in range(n):
##    plt.figure()
##    plt.plot(piLocalV[0,m,i,:])
###
#plt.show()
#
#
#print(np.average(piAverageV_avg_runs))
#
#











# Create a AIS calculator and run it:
aisCalcClass = JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
aisCalc = aisCalcClass()
aisCalc.setProperty("NORMALISE", "true") # Normalise the individual variables

aisCalc.setProperty("k", "4") # Use Kraskov parameter K=4 for 4 nearest points

# chemical V
aisAverageV = np.zeros((runs,size,size))
aisLocalV = np.zeros((runs,iterations-starting_point_time_series,size,size))

# chemical U
aisAverageU = np.zeros((runs,size,size))
aisLocalU = np.zeros((runs,iterations-starting_point_time_series,size,size))


for k in range(runs):
#    print(k)
    for i in range(size):
        for j in range(size):
            print(k, ' - ', i, ' - ', j)
            
            # PI calculator - V chemical
            sourceArray = v_hist[0,:,i,j].tolist()            # entire grid
#            sourceArray = frame_spot_histV[0,starting_point_time_series:,i,j].tolist()    # fixed frame around moving agent
            aisCalc.initialise(10) # Use history length 1 (Schreiber k=1)
            aisCalc.setObservations(JArray(JDouble, 1)(sourceArray))
            aisLocalV[k,:,i,j] = aisCalc.computeLocalOfPreviousObservations()
            aisAverageV[k,i,j] = aisCalc.computeAverageLocalOfObservations()
            
            # PI calculator - U chemical
#            sourceArray = u_hist[0,starting_point_time_series:,i,j].tolist()            # entire grid
##            sourceArray = frame_spot_histU[0,starting_point_time_series:,i,j].tolist()    # fixed frame around moving agent
#            piCalc.initialise(1) # Use history length 1 (Schreiber k=1)
#            piCalc.setObservations(JArray(JDouble, 1)(sourceArray))
##            piLocalU[k,i,j,:] = piCalc.computeLocalOfPreviousObservations()
#            piAverageU[k,i,j] = piCalc.computeAverageLocalOfObservations()

# chemical V
aisAverageV_avg_runs = np.average(aisAverageV, axis=0)
aisLocalV_avg_runs = np.average(aisLocalV, axis=0)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
x = range(size)
y = range(size)
x, y = np.meshgrid(x, y)
#ax = plt.contourf(x, y, piAverageV_avg_runs, 50, cmap=cm.jet)           # average
surf = ax.plot_surface(x, y, aisAverageV_avg_runs, cmap=cm.jet, linewidth=0)
plt.title('Average Predictive Information')
#cbar = fig.colorbar(ax, orientation='vertical')
#cbar.set_clim(2.0, 7.0)


plt.ion()
fig2 = plt.figure()
for i in range(iterations):
    if i % 100 == 0:
        print(i)
        plt.clf()
#        plt.contourf(x, y, piLocalV_avg_runs[i,:,:], 25, cmap=cm.jet)
        ais = plt.imshow(aisLocalV_avg_runs[i,:,:], interpolation='bicubic', cmap=cm.jet)
        ais.set_clim(vmin = np.min(aisLocalV_avg_runs[i,:,:]), vmax = np.max(aisLocalV_avg_runs[i,:,:]))
#        cbar = plt.colorbar();
#        cbar.set_clim(vmin = np.min(piLocalV_avg_runs[i,:,:]), vmax = np.max(piLocalV_avg_runs[i,:,:]))
#        plt.title('Predictive Information, i = ', i)
        plt.show()
        plt.pause(.01)

plt.ioff()

plt.show()












#
#
#
#
#### transfer entropy
#
## Create a PI calculator and run it:
#teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
#teCalc = teCalcClass()
#teCalc.setProperty("NORMALISE", "true") # Normalise the individual variables
#
#teCalc.setProperty("k", "4") # Use Kraskov parameter K=4 for 4 nearest points
#
## Set properties for auto-embedding of both source and destination
##  using the Ragwitz criteria:
##  a. Auto-embedding method
##teCalc.setProperty(teCalcClass.PROP_AUTO_EMBED_METHOD,
##		teCalcClass.AUTO_EMBED_METHOD_RAGWITZ)
###  b. Search range for embedding dimension (k) and delay (tau)
##teCalc.setProperty(teCalcClass.PROP_K_SEARCH_MAX, "40")
##teCalc.setProperty(teCalcClass.PROP_TAU_SEARCH_MAX, "40")
### Since we're auto-embedding, no need to supply k, l, k_tau, l_tau here:
##teCalc.initialise()
#
## chemical V
#teAverageV = np.zeros((runs,size,size))
#teLocalV = np.zeros((runs,iterations-starting_point_time_series,size,size))
#
## chemical U
#teAverageU = np.zeros((runs,size,size))
#teLocalU = np.zeros((runs,iterations-starting_point_time_series,size,size))
##sourceArray = np.zeros(iterations)
#for k in range(runs):
##    print(k)
#    for i in range(size):
#        for j in range(size):
#            print(k, ' - ', i, ' - ', j)
#            
#            # TE calculator - V chemical
#            destArray = v_hist[0,starting_point_time_series:,i,j].tolist()            # entire grid
##            destArray = frame_spot_histV[0,starting_point_time_series:,i,j].tolist()    # fixed frame around moving agent
#            
#            sourceArray = v_hist[0,starting_point_time_series:,m,m].tolist()            # entire grid
##            sourceArray = frame_spot_histV[0,starting_point_time_series:,m,m].tolist()    # fixed frame around moving agent
#            
#            teCalc.initialise(20)
##            print(teCalc.k, teCalc.k_tau)
#            teCalc.setObservations(JArray(JDouble, 1)(sourceArray), JArray(JDouble, 1)(destArray))
#            teLocalV[k,:,i,j] = teCalc.computeLocalOfPreviousObservations()
#            teAverageV[k,i,j] = teCalc.computeAverageLocalOfObservations()
#            
##            # TE calculator - U chemical
###            sourceArray = u_hist[0,:,i,j].tolist()
##            sourceArray = frame_spot_histU[0,starting_point_time_series:,i,j].tolist()
##            piCalc.initialise(1) # Use history length 1 (Schreiber k=1)
##            piCalc.setObservations(JArray(JDouble, 1)(sourceArray))
###            piLocalU[k,i,j,:] = piCalc.computeLocalOfPreviousObservations()
##            piAverageU[k,i,j] = piCalc.computeAverageLocalOfObservations()
#
## chemical V
#teAverageV_avg_runs = np.average(teAverageV, axis=0)
#teLocalV_avg_runs = np.average(teLocalV, axis=0)
#fig3 = plt.figure()
#ax3 = fig3.add_subplot(111, projection='3d')
#
## Plot the surface
#x = range(size)
#y = range(size)
#x, y = np.meshgrid(x, y)
##ax.plot_surface(x, y, piAverageV_avg, cmap=cm.jet)
##plt.imshow(piAverageV_avg, interpolation='bilinear')
##ax3 = plt.contourf(x, y, teAverageV_avg_runs, 50, cmap=cm.jet)
#surf = ax3.plot_surface(x, y, teAverageV_avg_runs, cmap=cm.jet, linewidth=0)
##plt.title('Average Transfer Entropy')
##plt.clim(2.0,3.0)
#
#
#plt.ion()
#fig4 = plt.figure()
#for i in range(iterations):
#    if i % 100 == 0:
#        print(i)
#        plt.clf()
##        plt.contourf(x, y, piLocalV_avg_runs[i,:,:], 25, cmap=cm.jet)
#        te = plt.imshow(teLocalV_avg_runs[i,:,:], interpolation='bicubic', cmap=cm.jet)
#        te.set_clim(vmin = np.min(teLocalV_avg_runs[i,:,:]), vmax = np.max(teLocalV_avg_runs[i,:,:]))
##        cbar = plt.colorbar();
##        cbar.set_clim(vmin = np.min(teLocalV_avg_runs[i,:,:]), vmax = np.max(teLocalV_avg_runs[i,:,:]))
##        plt.title('Predictive Information, i = ', i)
#        plt.show()
#        plt.pause(.01)
#
#plt.ioff()
#
#
#
#plt.show()


