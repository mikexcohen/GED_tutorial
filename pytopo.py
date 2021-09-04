import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
# import scipy.io as sio
# from scipy import interpolate
from scipy.interpolate import griddata

def topoplotIndie(Values,chanlocs,title='',ax=0):

    ## import and convert channel locations from EEG structure
    labels = []
    Th     = []
    Rd     = []
    x      = []
    y      = []

    #
    for ci in range(len(chanlocs[0])):
        labels.append(chanlocs[0]['labels'][ci][0])
        Th.append(np.pi/180*chanlocs[0]['theta'][ci][0][0])
        Rd.append(chanlocs[0]['radius'][ci][0][0])
        x.append( Rd[ci]*np.cos(Th[ci]) )
        y.append( Rd[ci]*np.sin(Th[ci]) )



    ## remove infinite and NaN values
    # ...
    
    # plotting factors
    headrad = .5
    plotrad = .6

    # squeeze coords into head
    squeezefac = headrad/plotrad
    # to plot all inside the head cartoon
    x = np.array(x)*squeezefac
    y = np.array(y)*squeezefac


    ## create grid
    xmin = np.min( [-headrad,np.min(x)] )
    xmax = np.max( [ headrad,np.max(x)] )
    ymin = np.min( [-headrad,np.min(y)] )
    ymax = np.max( [ headrad,np.max(y)] )
    xi   = np.linspace(xmin,xmax,67)
    yi   = np.linspace(ymin,ymax,67)

    # spatially interpolated data
    Xi, Yi = np.mgrid[xmin:xmax:67j,ymin:ymax:67j]
    Zi = griddata(np.array([y,x]).T,Values,(Yi,Xi))
#     f  = interpolate.interp2d(y,x,Values)
#     Zi = f(yi,xi)

    ## Mask out data outside the head
    mask = np.sqrt(Xi**2 + Yi**2) <= headrad
    Zi[mask == 0] = np.nan


    ## create topography
    # make figure
    if ax==0:
        fig  = plt.figure()
        ax   = fig.add_subplot(111, aspect = 1)
    clim = np.max(np.abs(Zi[np.isfinite(Zi)]))*.8
    ax.contourf(yi,xi,Zi,60,cmap=plt.cm.jet,zorder=1, vmin=-clim,vmax=clim)

    # head ring
    circle = patches.Circle(xy=[0,0],radius=headrad,edgecolor='k',facecolor='w',zorder=0)
    ax.add_patch(circle)

    # ears
    circle = patches.Ellipse(xy=[np.min(xi),0],width=.05,height=.2,angle=0,edgecolor='k',facecolor='w',zorder=-1)
    ax.add_patch(circle)
    circle = patches.Ellipse(xy=[np.max(xi),0],width=.05,height=.2,angle=0,edgecolor='k',facecolor='w',zorder=-1)
    ax.add_patch(circle)

    # nose (top, left, right)
    xy = [[0,np.max(yi)+.06], [-.2,.2],[.2,.2]]
    polygon = patches.Polygon(xy=xy,facecolor='w',edgecolor='k',zorder=-1)
    ax.add_patch(polygon)
    
    
    # add the electrode markers
    ax.scatter(y,x,marker='o', c='k', s=15, zorder = 3)

    ax.set_xlim([-.6,.6])
    ax.set_ylim([-.6,.6])
    ax.axis('off')
    ax.set_title(title)
    ax.set_aspect('equal')
#     plt.show()
