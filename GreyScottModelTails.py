#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 19:42:32 2017

Grey-Scott model found online, details below
Introducing now two different types of autocatalysts (see Virgo's thesis)

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

plt.close('all')


# Parameters from http://www.aliensaint.com/uo/java/rd/
# -----------------------------------------------------
n  = 200
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
Du, Dv, F, k = 0.16, 0.08, 0.04, 0.065 #
Du, Dv, Ds, F, k_v, k_s, alpha_s = 0.16, 0.08, 0.0, 0.04, 0.105, 0.0, 0. #
Du, Dv, Ds, F, k_v, k_s, alpha_s = 0.16, 0.08, 0.008, 0.0347, 0.2, 0.005, 0.8 #



Z = np.zeros((n+2,n+2), [('U', np.double), ('V', np.double), ('S', np.double)])
U,V,S = Z['U'], Z['V'], Z['S']
u,v,s = U[1:-1,1:-1], V[1:-1,1:-1], S[1:-1,1:-1]

r = 20
r2 = 14
u[...] = 1.0
m = int(n/2)
U[m-r:m+r,m-r:m+r] = 0.50
V[m-r:m+r,m-r:m+r] = 0.25
S[m-r2:m+r2,m+r:m+r+2*r2] = .50
u += 0.05*np.random.random((n,n))
v += 0.05*np.random.random((n,n))
s += 0.05*np.random.random((n,n))

a = u[u>0]
aa = v[v>0]
aaa = s[s>0]

plt.ion()

size = 4 * np.array(Z.shape)
dpi = 72.0
figsize= size[1]/float(dpi),size[0]/float(dpi)
fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
plt.subplot(211)
im = plt.imshow(V, interpolation='bicubic', cmap=plt.cm.gray_r)
plt.subplot(212)
im2 = plt.imshow(S, interpolation='bicubic', cmap=plt.cm.gray_r)
plt.xticks([]), plt.yticks([])


for i in range(25000):
    print(i)
    Lu = (                 U[0:-2,1:-1] +
          U[1:-1,0:-2] - 4*U[1:-1,1:-1] + U[1:-1,2:] +
                           U[2:  ,1:-1] )
    Lv = (                 V[0:-2,1:-1] +
          V[1:-1,0:-2] - 4*V[1:-1,1:-1] + V[1:-1,2:] +
                           V[2:  ,1:-1] )
    Ls = (                 S[0:-2,1:-1] +
          S[1:-1,0:-2] - 4*S[1:-1,1:-1] + S[1:-1,2:] +
                           S[2:  ,1:-1] )

    uvv = u*v*v
    vss = v*s*s
    u += (Du*Lu - uvv +  F*(1.02-u))
    v += (Dv*Lv + uvv - alpha_s*vss - k_v*v)
    s += (Ds*Ls + alpha_s*vss - k_s*s)

    if i % 100 == 0:
        im.set_data(V)
        im.set_clim(vmin=V.min(), vmax=V.max())
        
        im2.set_data(S)
        im2.set_clim(vmin=S.min(), vmax=S.max())

        plt.draw()
        plt.pause(.01)
#         To make movie
#        plt.savefig("./tmp-%03d.png" % (i/10) ,dpi=dpi)

plt.ioff()
# plt.savefig("../figures/zebra.png",dpi=dpi)
# plt.savefig("../figures/bacteria.png",dpi=dpi)
# plt.savefig("../figures/fingerprint.png",dpi=dpi)
plt.show()
