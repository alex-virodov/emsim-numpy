#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import scipy as sp
import scipy.special
import numpy.linalg
#import matplotlib.animation as animation

# enable debugging
import cgitb
cgitb.enable(logdir='./cgitblog')

import os
import os.path
import logging
import cgi


# Enable for profiling
# from profilehooks import profile, timecall
def timecall(f): return f

plotting = True
if (plotting): import matplotlib.pyplot as plt


@timecall
def solverEFIE_piecewise(geom, f_einc, wavek):
    """
        Solve the frequency-stationary EFIE equations using piecewise basis functions.
        geom         - a list of contours, where each contour is a list of points
        f_einc(x, y) - a function that would return the complex incident electric 
                       field for a given point array.
        
        returns (J, xj, yj, L, cond, f_scat, f_rcs)
        J           - the list  of complex currents at points xj, yj
        cond        - integral matrix condition number
        f_scat(x,y) - function that returns the complex value of the scattered field
                      at points x,y.
        f_rcs(phi)  - function that computes the RCS at angle phi.
        
        reference: http://onlinelibrary.wiley.com/doi/10.1002/9780470495094.app3/pdf
    """
    eta = 377
    c = (eta * wavek / 4)
    
    x, y  = (np.array([]),np.array([]))
    xc,yc = (np.array([]),np.array([]))
    L = np.array([])
    
    # Can "flatten" geometry into one long list because this approximation
    # considers only centers/length of individual line segments. So it doesn't 
    # really matter if each segment is joined by ends with another or not.
    
    # Note: can't just flatten the geometry at the beginning because
    # contours are disconnected, and flattening would connect them, introducing
    # new segments.    
    
    for g in geom:
        # Get x,y points and the rx,ry which are the same points
        # with index offset by 1 (aka next point lands at the same index)
        tx,ty = np.array(zip(*g))
        rx,ry = np.roll((tx,ty), 1, axis=1)
        
        # The current will be evaluated at centers of line segments
        txc = (tx + rx)/2
        tyc = (ty + ry)/2
    
        tL  = np.sqrt( (rx - tx)**2 + (ry - ty)**2 )
    
        # TODO: there must be a better python/numpy way to do this.
        x  = np.concatenate((x, tx))
        y  = np.concatenate((y, ty))
        xc = np.concatenate((xc, txc))
        yc = np.concatenate((yc, tyc))
        L  = np.concatenate((L, tL))
    

    # Number of elements equals number of lengths (across all contours)    
    N = len(L)

    Z_mn = np.zeros((N,N))*0j

    # Build the integral matrix using the approximations in the paper
    for m in range(N):
        for n in range(N):
            if (m == n):
                # 0.455 = (e^gamma)/2
                Z_mn[m][n] = c * L[n] * (1 - (1j * (2/np.pi) * np.log(0.455 * wavek*L[n])-1))
            else:
                R = np.sqrt((xc[m]-xc[n])**2 + (yc[m]-yc[n])**2)
                Z_mn[m][n] = c * L[n] * scipy.special.hankel2(0, wavek*R)
    
    V_m = f_einc(xc, yc)
    
    # Compute the inverse of the integral matrix
    inv_Z_mn = numpy.linalg.inv(Z_mn)
    J = inv_Z_mn.dot(V_m)
    
    # Compute condition number
    cond = numpy.linalg.norm(Z_mn, ord=1) * numpy.linalg.norm(inv_Z_mn, ord=1) 

    # RCS Impedance matrix
    # Note: using numpy broadcasting here to multiply each row by L
    Imp_mn = numpy.linalg.inv(Z_mn * L)

    # RCS Voltage matrix
    Vi_m = L * f_einc(x, y)

    
    # Make a function that computes the scattered field using the
    # approximations above
    # TODO: vectorize arguments
    def f_scat(u,v):
        R = np.sqrt((u-xc)**2 + (v-yc)**2)
        return np.sum(-c * J * L * scipy.special.hankel2(0, wavek*R))
    
    # Make a function that computes the RCS using the
    # approximations above
    # TODO: Definition correct only up to scale factor, make proper dB.
    def f_rcs(phi):
        # Scattering matrix for angle phi
        Vs_n = np.exp(1j * wavek * (x * np.cos(phi) + y * np.sin(phi))) 
        return np.log(eta * c * np.abs((Vs_n.dot(Imp_mn).dot(Vi_m)))**2)
    
    return (J, xc, yc, cond, f_scat, f_rcs) 
    
    


def makeCircle(n, cx, cy, phase=0):
    """
        Make a list of (x,y) points that correspond to a circle located
        at cx, cy. First point is not repeated.
        n     - number of points
        cx,cy - circle center
        phase - rotate the circle by this multiple of angluar distance between
                approximating points. phase=0.5 will make the first edge to
                be parallel to y-axis for example.
                
        Returns [(x,y)]
    """
    theta = np.linspace(0, 2 * sp.pi, n, endpoint=False)
    theta += theta[1] * phase
    
    x = np.cos(theta) + cx
    y = np.sin(theta) + cy
    
    # Transpose result into list of points
    return zip(x,y)


def makeGeometry(numx, numy, cx, cy, gvx, gvy, n, phase, center=False):
    """
        Make an array of spheres starting at (cx,cy) and generated by
        vector (gvx,gvy) and the tangential vector (gvy,gvx).
        n, phase are arguments passed to makeCircle().
        
        Returns [contour] = [ [(x,y)] ] 
    """
    
    c   = np.array([cx,  cy])
    gv  = np.array([gvy, gvx])
    gvt = np.array([gvx, gvy])
    geom = []
    
    dc = (c + gv*(numx-1) + gvt*(numy-1))/2
    
    for i in range(numx):
        for j in range(numy):
            r = c + i * gv + j * gvt - dc
            geom.append(makeCircle(n, r[0], r[1],phase))
            
    return geom

@timecall
def makeField(N, w, f_einc, f_scat, showIncidental, showScattered, showIncidentalHalf):
    """
        Make the numerical representation of the field on (-w,w)x(-w,w)
        domain discretized into NxN points.
        
        Returns (A,xo,yo)
        A  - the complex field values
        xo - x-positions array (by meshgrid)
        yo - y-positions array (by meshgrid)
        
    """
    (xo, yo) = np.meshgrid(  np.linspace(-w, w, N),
                             np.linspace(-w, w, N))
    
    A = np.zeros((N,N)) * 0j
    
    if (showIncidental):                
        A = f_einc(xo, yo)
    
    if (showScattered):
        # TODO: Fix huge slowdown here. Profiler says 0.7 seconds.
        for ix in range(N):
            for iy in range(N):
                A[ix][iy] += f_scat(xo[ix][iy], yo[ix][iy])
                
    if (showIncidentalHalf):
        pass
        A[:][:N/2] += f_einc(xo, yo)[:][:N/2]
        

    return (A, xo, yo)
    
@timecall
def plotProblem(frame, A, xo, yo, geom, w, frames, cond, showCurrent=False):
    phase =  np.exp(1j * (float(frame)/float(frames))*2*np.pi)
    B = A * phase
    plt.cla()
    plt.pcolor(xo, yo, B.real, cmap='gray')
    for g in geom:
        (x,y) = zip(*g)
        x = np.concatenate((x, [x[0]]), axis=1)
        y = np.concatenate((y, [y[0]]), axis=1)
        plt.plot(x, y, 'g', linewidth=4)
        
    plt.text(-w+0.1, -w+0.1+(w)*0.2, 'frame=%.0f' % frame, backgroundcolor='w')
    plt.text(-w+0.1, -w+0.1,         'condition number:%0.1f' % cond, backgroundcolor='w')
    
    # TODO: Broken for multiple objects. Fix.
    #if (showCurrent):
        # jscalar = (I*phase).real*2000
        # plt.quiver(
        #    xc, yc, 
        #    (x[0:N]-xc)*jscalar, (y[0:N]-yc)*jscalar, 
        #    color='b', units='inches', scale=1.0)


def main(wavek, fieldn, geom, save, doAnimation, fileprefix, frames):
    np.set_printoptions(suppress =True, linewidth =200)
    
    wavelen = 2*sp.pi / wavek;
    
    def Einc(x, y, phi=np.pi): return np.exp(1j * wavek * (x * np.cos(phi) + y * np.sin(phi)))
    
    # Geometry initialization
    #circle1 = makeCircle(20, cx=  0, cy=-10, phase=0.5)
    #circle2 = makeCircle(20, cx=  0, cy=+10, phase=0.5)
    #geom = [circle1, circle2]
        
    _,_,_,cond,f_scat,f_rcs = solverEFIE_piecewise(geom, Einc,  wavek)
    # print(I)
    print("condition number:", cond)
    
    # Done here if plotting is disabled
    if (not plotting): return
    
    w=8*wavelen

    (A, xo, yo) = makeField(fieldn, w, Einc, f_scat, 
                    showIncidental=False, showScattered=True, showIncidentalHalf=True)
    # print(A)

    def animate(i):
        plotProblem(i, A, xo, yo, geom, w, frames, cond, showCurrent=False)
        return []
    
    if (doAnimation):
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-w, w), ylim=(-w, w), aspect='equal')
        
        if (save):
            fig = plt.figure(figsize=(5,5))
            for i in range(frames):
                plt.clf()
                animate(i)
                plt.savefig(fileprefix + '.%d.png' % i)
        else:
            ani = animation.FuncAnimation(fig, animate, frames, interval=250)
            
                
    else:
        phi = np.linspace(0, 2 * sp.pi, 120)
        rcs_phi = np.zeros(phi.shape[0])
        for n in range(phi.shape[0]): rcs_phi[n] = f_rcs(phi[n])
        
        if (save):
            fig = plt.figure(figsize=(5,5))
            animate(0)

            logging.debug('making:' + fileprefix + '-field.png');
            plt.savefig(fileprefix + '-field.png')
            
            plt.clf()
            fig.add_subplot(111, autoscale_on=False, xlim=(0, 2 * sp.pi), ylim=(np.min(rcs_phi)-0.5, np.max(rcs_phi)+0.5))
            plt.plot(phi, rcs_phi.real)
            logging.debug('making:' + fileprefix + '-rcs.png');
            plt.savefig(fileprefix + '-rcs.png')
            
        else:
            plt.figure(figsize=(20,10))
            ax = plt.subplot(1,2,1, aspect='equal')
            animate(0)

            ax = plt.subplot(1,2,2)
            plt.plot(phi, rcs_phi.real)
        
            


    # Show the plot
    if (not save):
        plt.show()

if __name__ == "__main__":
    
#    geom = makeGeometry(numx=1, numy=1, cx=0, cy=0, gvx=0, gvy=10, n=20, phase=0.5, center=True)
#    main(geom, save=True, doAnimation=False, fileprefix='prefix', frames=6)    
	logging.basicConfig(filename='code.log',level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
	logging.info('starting')

	print("Content-Type: text/plain;charset=utf-8")
	print("")
	print("start")

	args = cgi.FieldStorage()

	for arg in args: logging.debug('argument:' + str(arg) + ':' + args[arg].value.strip())

	# Make sure prefix is numeric and doesn't contain 'cd / && rm -rf *' or similar
	prefix = '../cache/' + str(int(args['prefix'].value.strip()))
	logging.debug('prefix:[' + prefix + ']');

	# Make the geometry and the images
	wavek    = float(args['wavek'   ].value) / 100.0
	segments = int(  args['segments'].value)
	phase    = float(args['phase'   ].value) / 10.0
	fieldn   = int(  args['fieldn'  ].value)

	numx     = int(  args['gridnx'  ].value)
	numy     = int(  args['gridny'  ].value)


	if wavek < 0.0001: wavek = 0.5
    	wavek   = 2*sp.pi * wavek

	geom = makeGeometry(numx, numy, cx=0, cy=0, gvx=0, gvy=10, n=segments, phase=phase, center=True)
	main(wavek, fieldn, geom, save=True, doAnimation=False, fileprefix=prefix, frames=6)

	logging.info('done')
	print("done")


    
    

    




