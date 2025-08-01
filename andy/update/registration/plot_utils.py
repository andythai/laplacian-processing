"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
Atchuth Naveen
Code developed at UC Irvine.

Contains utility functions to genereate  plots for visualizing results during registration.
"""

import numpy as np
import matplotlib.pyplot as plt
from registration.reg_utils import *
import plotly.express as px
import plotly.graph_objects as go



def plot_samples(image3D, axis=0, sample_size=3, samples =[]):
    if len(samples)==0:
        samples = np.random.randint(0, image3D.shape[axis], sample_size)
    else:
        sample_size = len(samples)
    
    fig, ax = plt.subplots(nrows=int(np.ceil(sample_size/3)), ncols=3, figsize=(10, 5), squeeze = False)
    for idx, s in enumerate(samples):
        ax[int(idx/3)][idx%3].imshow(np.rollaxis(image3D, axis)[s])
        ax[int(idx/3)][idx%3].set_title("Layer:{}".format(s))
        
def get_normal_endpoints(points, normals):
    x = np.vstack([points.T[0], points.T[0]+normals.T[0]])
    y = np.vstack([points.T[1], points.T[1]+normals.T[1]])
    z = np.vstack([points.T[2], points.T[2]+normals.T[2]])
    
    return x,y,z

def get_lines( points, normals):
    pnx , pny, pnz = get_normal_endpoints(points,normals)
    
    x_lines = list()
    y_lines = list()
    z_lines = list()


    #create the coordinate list for the lines
    for s in range(pnx.T.shape[0]):
        for i in range(2):
            s=int(s)
            x_lines.append(pnx.T[s][i])
            y_lines.append(pny.T[s][i])
            z_lines.append(pnz.T[s][i])
            
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
    return x_lines,y_lines,z_lines


def plot2DCorrespondences(dataImage, templateImage, title=None, nheight =1):
    """
    dataimage = np.take(mdata, sno, axis=axis)
    templateimage = np.take(fdata, msno, axis=axis)
    """
    fedge, medge, fbin, mbin = getContours(dataImage, templateImage)
    
    mpoints, mnormals = estimate2Dnormals(np.array(medge.nonzero()).T, mbin)
    fpoints, fnormals = estimate2Dnormals(np.array(fedge.nonzero()).T, fbin)

    fc, mc = get2DCorrespondences(fedge, medge, fbin, mbin)
    
    mpoints = np.hstack([mpoints, np.ones( (mpoints.shape[0],1))])
    mnormals = np.hstack([mnormals, np.zeros( (mnormals.shape[0],1))])
    fpoints = np.hstack([fpoints, nheight*np.ones( (fpoints.shape[0],1))])
    fnormals = np.hstack([fnormals, np.zeros( (fnormals.shape[0],1))])
    
    
    fc = np.hstack([fc, nheight*np.ones( (fc.shape[0],1))])
    mc = np.hstack([mc, np.ones( (mc.shape[0],1))])

    cx_lines,cy_lines,cz_lines = get_correspondence_lines(mc, fc)
    mx_lines,my_lines,mz_lines = get_lines(mpoints,5* mnormals)
    fx_lines,fy_lines, fz_lines = get_lines(fpoints,5* fnormals)

    fig = go.Figure(data=[go.Scatter3d(x=mpoints[:,0], 
                                       y=mpoints[:,1], 
                                       z=mpoints[:,2],
                                       mode='markers', 
                                       name = "Moving Image")])

    fig.add_trace(go.Scatter3d(x=fpoints[:,0], 
                               y=fpoints[:,1], 
                               z=fpoints[:,2],
                               mode='markers', 
                               name = "Fixed Image"))

    fig.add_trace(go.Scatter3d(x=cx_lines,
                               y=cy_lines,
                               z=cz_lines,
                               mode='lines',
                               name = "Correspondences"))

    fig.add_trace(go.Scatter3d(x=mx_lines,
                               y=my_lines,
                               z=mz_lines,
                               mode='lines',
                               name ="Moving Image Normals"))
    fig.add_trace(go.Scatter3d(x=fx_lines,
                               y=fy_lines,
                               z=fz_lines,
                               mode='lines',
                               name ="Fixed Image Normals"))
    if title is None:
        title="2D Corrspondences"

    fig.update_traces(marker=dict(size=1))

    fig.update_layout(title={
        'text': title,
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'}, autosize=True) 

    fig.update_layout(scene=dict(xaxis=dict(showbackground=False, visible=False,
                                            showgrid=False),
                                 yaxis=dict(showbackground=False, visible=False),
                                 zaxis=dict(showbackground=False, visible=False),
                                 bgcolor='black'),
                                 plot_bgcolor='rgb(0,0,0)',)
    
    fig.update_traces(marker=dict(size=2),
                      selector=dict(mode='markers'))
    
    fig.show(renderer='browser')
    return fedge, medge

def get_correspondence_lines(source, target, correspondences= None):
    
    x_lines = list()
    y_lines = list()
    z_lines = list()

    if correspondences == None:
        assert source.shape==target.shape
        correspondences = np.arange(source.shape[0])

    #create the coordinate list for the lines
    for index, correspondence in enumerate(correspondences):
        if(correspondence ==-1):
            continue
        x_lines.append(source[index][0])
        y_lines.append(source[index][1])
        z_lines.append(source[index][2])
        
        x_lines.append(target[correspondence][0])
        y_lines.append(target[correspondence][1])
        z_lines.append(target[correspondence][2])
        
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
        
    return x_lines, y_lines, z_lines

def sample_correspondence_lines(source, target, correspondences, samples):
    x_lines = list()
    y_lines = list()
    z_lines = list()


    #create the coordinate list for the lines
    for  i,index in enumerate(samples):
        
        correspondence =correspondences[i]
        if(correspondence ==-1):
            continue
        x_lines.append(source[index][0])
        y_lines.append(source[index][1])
        z_lines.append(source[index][2])
        
        x_lines.append(target[correspondence][0])
        y_lines.append(target[correspondence][1])
        z_lines.append(target[correspondence][2])
        
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)
        
    return x_lines, y_lines, z_lines
    
        