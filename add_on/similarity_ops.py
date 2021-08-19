import numpy as np
from numpy import matlib as mb # matlib must be imported separately
from numpy.linalg import matrix_rank
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance
import torch
import scipy

def compute_spatial_similarity(conv1,conv2):
    """
    
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    im_similarity = np.zeros((conv1_normed.shape[0],conv1_normed.shape[0]))
    for zz in range(conv1_normed.shape[0]):
        repPx = mb.repmat(conv1_normed[zz,:],conv1_normed.shape[0],1)
        im_similarity[zz,:] = np.multiply(repPx,conv2_normed).sum(axis=1)
    similarity1 = np.reshape(np.sum(im_similarity,axis=1),out_sz)
    similarity2 = np.reshape(np.sum(im_similarity,axis=0),out_sz)
    return similarity1, similarity2 

def compute_spatial_similarity_fc(conv1,conv2,weighs,feat1,feat2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    load1 = np.matmul(feat1,weighs)
    load2 = np.matmul(feat2,weighs)
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
#     conv1_normed = conv1
#     conv2_normed = conv2
    repPx1 = mb.repmat(load2,conv2_normed.shape[0],1)
    repPx2 = mb.repmat(load1,conv2_normed.shape[0],1)
    im_similarity1 = np.multiply(repPx1,conv1_normed).sum(axis=1)
    im_similarity2 = np.multiply(repPx2,conv2_normed).sum(axis=1)
    res1 = np.reshape(im_similarity1,out_sz)
    res2 = np.reshape(im_similarity2,out_sz)
    sim1 = res1.copy()
    sim2 = res2.copy()
    dif1 = res1.copy()
    dif2 = res2.copy()
    sim1[sim1<0]=0
    sim2[sim2<0]=0
    dif1[dif1>0]=0
    dif2[dif2>0]=0
    dif1=-dif1
    dif2=-dif2
    return sim1, sim2, dif1, dif2

def compute_spatial_similarity_B(conv1,conv2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    pool1 = pool1/np.linalg.norm(pool1)
    pool2 = pool2/np.linalg.norm(pool2)
    dproduct = pool1*pool2
    posvector = dproduct.copy()
    posvector[posvector<0]=0
    negvector = dproduct.copy()
    negvector[negvector>0]=0
    negvector=-negvector
    repPx_pos = mb.repmat(posvector,conv2_normed.shape[0],1)
    repPx_neg = mb.repmat(negvector,conv2_normed.shape[0],1)
    sim1 = np.reshape(np.multiply(repPx_pos,conv1_normed).sum(axis=1),out_sz)
    sim2 = np.reshape(np.multiply(repPx_pos,conv2_normed).sum(axis=1),out_sz)
    dif1 = np.reshape(np.multiply(repPx_neg,conv1_normed).sum(axis=1),out_sz)
    dif2 = np.reshape(np.multiply(repPx_neg,conv2_normed).sum(axis=1),out_sz)
    return sim1, sim2, dif1, dif2

def compute_spatial_similarity_fc_B(conv1,conv2,weighs,feat1,feat2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    dproduct = feat1*feat2
#     posvector = dproduct.copy()
#     posvector[posvector<0]=0
#     negvector = dproduct.copy()
#     negvector[negvector>0]=0
#     negvector=-negvector
#     load_pos = np.matmul(posvector,weighs)
#     load_neg = np.matmul(negvector,weighs)
    load = np.matmul(dproduct,weighs)
    load_pos = load.copy()
    load_pos[load_pos<0]=0
    load_neg = load.copy()
    load_neg[load_neg>0]=0
    load_neg = -load_neg
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
#     conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
#     conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    conv1_normed = conv1
    conv2_normed = conv2
    repPx_pos = mb.repmat(load_pos,conv2_normed.shape[0],1)
    repPx_neg = mb.repmat(load_neg,conv2_normed.shape[0],1)
    sim1 = np.reshape(np.multiply(repPx_pos,conv1_normed).sum(axis=1),out_sz)
    sim2 = np.reshape(np.multiply(repPx_pos,conv2_normed).sum(axis=1),out_sz)
    dif1 = np.reshape(np.multiply(repPx_neg,conv1_normed).sum(axis=1),out_sz)
    dif2 = np.reshape(np.multiply(repPx_neg,conv2_normed).sum(axis=1),out_sz)
    return sim1, sim2, dif1, dif2

# def compute_spatial_similarity_fc_B_afc(conv1,conv2,weighs,feat1,feat2):
#     """
#     Takes in the last convolutional layer from two images, computes the pooled output
#     feature, and then generates the spatial similarity map for both images.
#     """
#     dproduct = feat1*feat2
#     posvector = dproduct.copy()
#     posvector[posvector<0]=0
#     negvector = dproduct.copy()
#     negvector[negvector>0]=0
#     load_pos = np.matmul(posvector,weighs)
#     load_neg = np.matmul(negvector,weighs)
#     print(load_pos,load_neg)
#     pool1 = np.mean(conv1,axis=0)
#     pool2 = np.mean(conv2,axis=0)
#     out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
# #     conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
# #     conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
#     conv1_normed = conv1
#     conv2_normed = conv2
#     repPx_pos = mb.repmat(load_pos,conv2_normed.shape[0],1)
#     repPx_neg = mb.repmat(load_neg,conv2_normed.shape[0],1)
#     sim1 = np.reshape(np.multiply(repPx_pos,conv1_normed).sum(axis=1),out_sz)
#     sim2 = np.reshape(np.multiply(repPx_pos,conv2_normed).sum(axis=1),out_sz)
#     dif1 = np.reshape(np.multiply(repPx_neg,conv1_normed).sum(axis=1),out_sz)
#     dif2 = np.reshape(np.multiply(repPx_neg,conv2_normed).sum(axis=1),out_sz)
#     return sim1, sim2, dif1, dif2

def compute_spatial_similarity_C(conv1,conv2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    im_similarity = np.zeros((conv1_normed.shape[0],conv2_normed.shape[0]))
    indep_score_1 = np.zeros(conv1_normed.shape[0])
    indep_score_2 = np.zeros(conv2_normed.shape[0])
    for zz in range(conv1_normed.shape[0]):
        repPx = mb.repmat(conv1_normed[zz,:],conv1_normed.shape[0],1)
        im_similarity[zz,:] = np.multiply(repPx,conv2_normed).sum(axis=1)
        vect_1 = conv1_normed[zz,:]
        vect_2 = conv2_normed[zz,:]
        
        temp_1,resid,rank,s = np.linalg.lstsq(conv2_normed.T,vect_1)
        proj_1 = np.matmul(conv2_normed.T, temp_1)
        null_1 = vect_1-proj_1
        
        temp_2,resid,rank,s = np.linalg.lstsq(conv1_normed.T,vect_2)
        proj_2 = np.matmul(conv1_normed.T, temp_2)
        null_2 = vect_2-proj_2
        
        indep_score_1[zz] = np.linalg.norm(null_1)/np.linalg.norm(vect_1)
        indep_score_2[zz] = np.linalg.norm(null_2)/np.linalg.norm(vect_2)
        
        
    res1 = np.reshape(np.sum(im_similarity,axis=1),out_sz)
    res2 = np.reshape(np.sum(im_similarity,axis=0),out_sz)
    sim1 = np.copy(res1)
    sim2 = np.copy(res2)
    dif1 = np.copy(res1)
    dif2 = np.copy(res1)
    sim1[sim1<0]=0
    sim2[sim2<0]=0
    dif1[dif1>0]=0
    dif2[dif2>0]=0
    dif1=-dif1
    dif2=-dif2
    
    ind1 = np.reshape(indep_score_1,out_sz)
    ind2 = np.reshape(indep_score_2,out_sz)
    return sim1, sim2, dif1, dif2, ind1, ind2    

def compute_spatial_similarity_fc_C(conv1,conv2,weighs):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    fc_map1 = np.matmul(conv1, weighs.T)
    fc_map2 = np.matmul(conv2, weighs.T)
    im_similarity = np.zeros((fc_map1.shape[0],fc_map2.shape[0]))
    indep_score_1 = np.zeros(fc_map1.shape[0])
    indep_score_2 = np.zeros(fc_map2.shape[0])
    for zz in range(fc_map1.shape[0]):
        repPx = mb.repmat(fc_map1[zz,:],fc_map1.shape[0],1)
        im_similarity[zz,:] = np.multiply(repPx,fc_map2).sum(axis=1)
        vect_1 = fc_map1[zz,:]
        vect_2 = fc_map2[zz,:]
        
        temp_1,resid,rank,s = np.linalg.lstsq(fc_map2.T,vect_1)
        proj_1 = np.matmul(fc_map2.T, temp_1)
        null_1 = vect_1-proj_1
        
        temp_2,resid,rank,s = np.linalg.lstsq(fc_map1.T,vect_2)
        proj_2 = np.matmul(fc_map1.T, temp_2)
        null_2 = vect_2-proj_2
        
        indep_score_1[zz] = np.linalg.norm(null_1)/np.linalg.norm(vect_1)
        indep_score_2[zz] = np.linalg.norm(null_2)/np.linalg.norm(vect_2)
    res1 = np.reshape(np.sum(im_similarity,axis=1),out_sz)
    res2 = np.reshape(np.sum(im_similarity,axis=0),out_sz)
    sim1 = np.copy(res1)
    sim2 = np.copy(res2)
    dif1 = np.copy(res1)
    dif2 = np.copy(res1)
    sim1[sim1<0]=0
    sim2[sim2<0]=0
    dif1[dif1>0]=0
    dif2[dif2>0]=0
    dif1=-dif1
    dif2=-dif2
    
    ind1 = np.reshape(indep_score_1,out_sz)
    ind2 = np.reshape(indep_score_2,out_sz)
    return sim1, sim2, dif1, dif2, ind1, ind2    

def compute_spatial_similarity_grad(conv1,conv2,weighs):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    shape = conv1.shape
    out_sz = (int(np.sqrt(shape[0])),int(np.sqrt(shape[0])))
    conv1 = np.reshape(conv1,(1,-1)).ravel()
    conv2 = np.reshape(conv2,(1,-1)).ravel()
    feat1 = np.matmul(conv1, weighs.T)
    feat2 = np.matmul(conv2, weighs.T)
    grad1 = np.matmul(weighs.T,feat1.T).ravel()
    grad2 = np.matmul(weighs.T,feat2.T).ravel()
    heatmap1 = conv1*grad1
    heatmap2 = conv2*grad2
    res1 = np.reshape(heatmap1,shape)
    res2 = np.reshape(heatmap2,shape)
    res1 = np.sum(res1, axis=1)
    res2 = np.sum(res2, axis=1)
    res1 = np.reshape(res1,out_sz)
    res2 = np.reshape(res2,out_sz)
    sim1 = np.copy(res1)
    sim2 = np.copy(res2)
    dif1 = np.copy(res1)
    dif2 = np.copy(res2)
    sim1[sim1<0]=0
    sim2[sim2<0]=0
    dif1[dif1>0]=0
    dif2[dif2>0]=0
    dif1=-dif1
    dif2=-dif2
    return sim1, sim2, dif1, dif2

def compute_spatial_similarity_L2_fc(conv1,conv2,weighs,feat1,feat2,L2_dis):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    feat1 = feat1/np.linalg.norm(feat1)
    feat2 = feat2/np.linalg.norm(feat2)
    load1diff = (feat1-feat2)/L2_dis
    #load2diff = (feat2-feat1)/L2_dis
    load2diff = (feat2-feat1)/L2_dis
    load1diff[feat1==0]=0
    load2diff[feat2==0]=0
    load1 = np.matmul(load1diff,weighs)
    load2 = np.matmul(load2diff,weighs)
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
#     conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
#     conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    
    conv1_normed = conv1
    conv2_normed = conv2
    repPx1 = mb.repmat(load1,conv1_normed.shape[0],1)
    repPx2 = mb.repmat(load2,conv2_normed.shape[0],1)
    im_similarity1 = np.multiply(repPx1,conv1_normed).sum(axis=1)
    im_similarity2 = np.multiply(repPx2,conv2_normed).sum(axis=1)
    res1 = np.reshape(im_similarity1,out_sz)
    res2 = np.reshape(im_similarity2,out_sz)
    sim1 = res1.copy()
    sim2 = res2.copy()
    dif1 = res1.copy()
    dif2 = res2.copy()
#     sim1 = -sim1
#     sim2 = -sim2
#     dif1 = -dif1
#     dif2 = -dif2
#     sim1[sim1<0]=0
#     sim2[sim2<0]=0
#     dif1[dif1>0]=0
#     dif2[dif2>0]=0
#     dif1=-dif1
#     dif2=-dif2
    return sim1, sim2, dif1, dif2

# def compute_spatial_similarity_L2_fc_B_afc(conv1,conv2,weighs,feat1,feat2,L2_dis):
#     """
#     Takes in the last convolutional layer from two images, computes the pooled output
#     feature, and then generates the spatial similarity map for both images.
#     """
#     feat1 = feat1/np.linalg.norm(feat1)
#     feat2 = feat2/np.linalg.norm(feat2)
#     load1diff = (feat1-feat2)/L2_dis
#     #load2diff = (feat2-feat1)/L2_dis
#     load2diff = (feat2-feat1)/L2_dis
#     load1diff[feat1==0]=0
#     load2diff[feat2==0]=0
#     pos1 = load1diff.copy()
#     pos1[pos1<0]==0
#     dif1 = load1diff.copy()
#     dif1[dif1>0]==0
#     #dif1 = -dif1
#     pos2 = load2diff.copy()
#     pos2[pos2<0]==0
#     dif2 = load2diff.copy()
#     dif2[dif2>0]==0
#     #dif2 = -dif2
#     load1pos = np.matmul(pos1,weighs)
#     load1dif = np.matmul(dif1,weighs)
#     load2pos = np.matmul(pos2,weighs)
#     load2dif = np.matmul(dif2,weighs)
#     pool1 = np.mean(conv1,axis=0)
#     pool2 = np.mean(conv2,axis=0)
#     out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
#     conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
#     conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    
#     posrepPx1 = mb.repmat(-load1pos,conv1_normed.shape[0],1)
#     posrepPx2 = mb.repmat(-load2pos,conv2_normed.shape[0],1)
#     im_similarity1 = np.multiply(posrepPx1,conv1_normed).sum(axis=1)
#     im_similarity2 = np.multiply(posrepPx2,conv2_normed).sum(axis=1)

#     difrepPx1 = mb.repmat(-load1dif,conv1_normed.shape[0],1)
#     difrepPx2 = mb.repmat(-load2dif,conv2_normed.shape[0],1)
#     im_difference1 = np.multiply(difrepPx1,conv1_normed).sum(axis=1)
#     im_difference2 = np.multiply(difrepPx2,conv2_normed).sum(axis=1)
    
#     sim1 = np.reshape(im_similarity1,out_sz)
#     sim2 = np.reshape(im_similarity2,out_sz)
#     dif1 = np.reshape(im_difference1,out_sz)
#     dif2 = np.reshape(im_difference2,out_sz)
#     return sim1, sim2, dif1, dif2

def compute_spatial_similarity_L2_fc_B(conv1,conv2,weighs,feat1,feat2,L2_dis):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    feat1 = feat1/np.linalg.norm(feat1)
    feat2 = feat2/np.linalg.norm(feat2)
    load1diff = (feat1-feat2)/L2_dis
    #load2diff = (feat2-feat1)/L2_dis
    load2diff = (feat2-feat1)/L2_dis
#     load1diff[feat1==0]=0
#     load2diff[feat2==0]=0
    
    load1 = np.matmul(load1diff,weighs)
    load2 = np.matmul(load2diff,weighs)
    load1pos = load1.copy()
    load1pos[load1pos<0]=0
    load1neg = load1.copy()
    load1neg[load1neg>0]=0
    load1neg = -load1neg
    load2pos = load2.copy()
    load2pos[load2pos<0]=0
    load2neg = load2.copy()
    load2neg[load2neg>0]=0
    load2neg = -load2neg
    
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    
    posrepPx1 = mb.repmat(load1pos,conv1_normed.shape[0],1)
    posrepPx2 = mb.repmat(load2pos,conv2_normed.shape[0],1)
    im_similarity1 = np.multiply(posrepPx1,conv1_normed).sum(axis=1)
    im_similarity2 = np.multiply(posrepPx2,conv2_normed).sum(axis=1)

    negrepPx1 = mb.repmat(load1neg,conv1_normed.shape[0],1)
    negrepPx2 = mb.repmat(load2neg,conv2_normed.shape[0],1)
    im_difference1 = np.multiply(negrepPx1,conv1_normed).sum(axis=1)
    im_difference2 = np.multiply(negrepPx2,conv2_normed).sum(axis=1)
    
    sim1 = np.reshape(im_similarity1,out_sz)
    sim2 = np.reshape(im_similarity2,out_sz)
    dif1 = np.reshape(im_difference1,out_sz)
    dif2 = np.reshape(im_difference2,out_sz)
    return sim1, sim2, dif1, dif2

def L2_fc_test(conv1,conv2,weighs,feat1,feat2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    hm1list = []
    hm2list = []
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    feat1 = feat1/np.linalg.norm(feat1)
    feat2 = feat2/np.linalg.norm(feat2)
    diff = np.absolute(feat1-feat2)
    sort_ind = np.argsort(-diff)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    for ind in sort_ind:
        A_weighs = diff[ind]*weighs[ind]
        repPx1 = mb.repmat(A_weighs,conv1.shape[0],1)
        im_similarity1 = np.multiply(repPx1,conv1_normed).sum(axis=1)
        res1 = np.reshape(im_similarity1,out_sz)

        repPx2 = mb.repmat(A_weighs,conv2.shape[0],1)
        im_similarity2 = np.multiply(repPx2,conv2_normed).sum(axis=1)
        res2 = np.reshape(im_similarity2,out_sz)
        hm1list.append(res1)
        hm2list.append(res2)
    return hm1list,hm2list


def compute_spatial_similarity_fc_B_l1(conv1,conv2,weighs,feat1,feat2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    dproduct2 = feat1-feat2
    dproduct1 = dproduct2
#     posvector = dproduct.copy()
#     posvector[posvector<0]=0
#     negvector = dproduct.copy()
#     negvector[negvector>0]=0
#     negvector=-negvector
#     load_pos = np.matmul(posvector,weighs)
#     load_neg = np.matmul(negvector,weighs)
    load1 = np.matmul(dproduct1,weighs)
    load2 = np.matmul(dproduct2,weighs)
    load_pos1 = load1.copy()
    load_pos1[load_pos1<0]=0
    load_neg1 = load1.copy()
    load_neg1[load_neg1>0]=0
    load_neg1 = -load_neg1
    load_pos2 = load2.copy()
    load_pos2[load_pos2<0]=0
    load_neg2 = load2.copy()
    load_neg2[load_neg2>0]=0
    load_neg2 = -load_neg2
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
#     conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
#     conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    conv1_normed = conv1
    conv2_normed = conv2
    repPx_pos1 = mb.repmat(load_pos1,conv2_normed.shape[0],1)
    repPx_neg1 = mb.repmat(load_neg1,conv2_normed.shape[0],1)
    repPx_pos2 = mb.repmat(load_pos2,conv2_normed.shape[0],1)
    repPx_neg2 = mb.repmat(load_neg2,conv2_normed.shape[0],1)
    sim1 = np.reshape(np.multiply(repPx_pos1,conv1_normed).sum(axis=1),out_sz)
    sim2 = np.reshape(np.multiply(repPx_pos2,conv2_normed).sum(axis=1),out_sz)
    dif1 = np.reshape(np.multiply(repPx_neg1,conv1_normed).sum(axis=1),out_sz)
    dif2 = np.reshape(np.multiply(repPx_neg2,conv2_normed).sum(axis=1),out_sz)
    return sim1, sim2, dif1, dif2




def compute_spatial_similarity_fc_B_maxpooling(conv1,conv2,weighs,feat1,feat2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    dproduct = feat1*feat2
#     posvector = dproduct.copy()
#     posvector[posvector<0]=0
#     negvector = dproduct.copy()
#     negvector[negvector>0]=0
#     negvector=-negvector
#     load_pos = np.matmul(posvector,weighs)
#     load_neg = np.matmul(negvector,weighs)
    load = np.matmul(dproduct,weighs)
    load_pos = load.copy()
    load_pos[load_pos<0]=0
    load_neg = load.copy()
    load_neg[load_neg>0]=0
    load_neg = -load_neg
    pool1 = np.max(conv1,axis=0)
    pool2 = np.max(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
#     conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
#     conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    for i in range(0,conv1.shape[1]):
        conv1[:,i][conv1[:,i]<pool1[i]]=0
        conv2[:,i][conv2[:,i]<pool2[i]]=0
    conv1_normed = conv1
    conv2_normed = conv2
    repPx_pos = mb.repmat(load_pos,conv2_normed.shape[0],1)
    repPx_neg = mb.repmat(load_neg,conv2_normed.shape[0],1)
    sim1 = np.reshape(np.multiply(repPx_pos,conv1_normed).sum(axis=1),out_sz)
    sim2 = np.reshape(np.multiply(repPx_pos,conv2_normed).sum(axis=1),out_sz)
    dif1 = np.reshape(np.multiply(repPx_neg,conv1_normed).sum(axis=1),out_sz)
    dif2 = np.reshape(np.multiply(repPx_neg,conv2_normed).sum(axis=1),out_sz)
    return sim1, sim2, dif1, dif2

def compute_spatial_similarity_fc_SVD(conv1,conv2,weighs,feat1,feat2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    u1, _, _ = np.linalg.svd(conv1.T, full_matrices=False)
    u2, _, _ = np.linalg.svd(conv2.T, full_matrices=False)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    conv1_normed = conv1
    conv2_normed = conv2
    r1 = mb.repmat(u1[:,0],conv2_normed.shape[0],1)
    g1 = mb.repmat(u1[:,1],conv2_normed.shape[0],1)
    b1 = mb.repmat(u1[:,2],conv2_normed.shape[0],1)
    r1res = np.reshape(np.multiply(r1,conv2_normed).sum(axis=1),out_sz)
    g1res = np.reshape(np.multiply(g1,conv2_normed).sum(axis=1),out_sz)
    b1res = np.reshape(np.multiply(b1,conv2_normed).sum(axis=1),out_sz)
    
    r2 = mb.repmat(u2[:,0],conv1_normed.shape[0],1)
    g2 = mb.repmat(u2[:,1],conv1_normed.shape[0],1)
    b2 = mb.repmat(u2[:,2],conv1_normed.shape[0],1)
    r2res = np.reshape(np.multiply(r2,conv1_normed).sum(axis=1),out_sz)
    g2res = np.reshape(np.multiply(g2,conv1_normed).sum(axis=1),out_sz)
    b2res = np.reshape(np.multiply(b2,conv1_normed).sum(axis=1),out_sz)
    
    res1 = np.array([r2res,g2res,b2res])
    res1 = (res1-res1.min())
    res1 = res1/res1.max()
    res2 = np.array([r1res,g1res,b1res])
    res2 = (res2-res2.min())
    res2 = res2/res2.max()
    res1 = np.ascontiguousarray(res1.transpose(1, 2, 0))*255
    res2 = np.ascontiguousarray(res2.transpose(1, 2, 0))*255
    return res1,res2

def combine_SVD(conv1,conv2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """    
    allconv = np.vstack((conv1,conv2))
    #allconv = (allconv.T/np.linalg.norm(allconv,axis=1)).T
    allconv = allconv-allconv.mean(axis=0)
    u, s, v = np.linalg.svd(allconv, full_matrices=False)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    u = u[:,0:3]*s[0:3]
    r1 = np.reshape(u[:,0][0:conv1.shape[0]],out_sz)
    g1 = np.reshape(u[:,1][0:conv1.shape[0]],out_sz)
    b1 = np.reshape(u[:,2][0:conv1.shape[0]],out_sz)
    r2 = np.reshape(u[:,0][conv1.shape[0]:],out_sz)
    g2 = np.reshape(u[:,1][conv1.shape[0]:],out_sz)
    b2 = np.reshape(u[:,2][conv1.shape[0]:],out_sz)

    r1 = 0.5+0.5*r1/(np.absolute(r1).max())
    g1 = 0.5+0.5*g1/(np.absolute(g1).max())
    b1 = 0.5+0.5*b1/(np.absolute(b1).max())
    res1 = np.array([r1,g1,b1])
#     res1 = np.absolute(res1)
    #print(res1.min())
#     res1 = (res1 - np.mean(res1)) / (np.std(res1))
#     res1 = res1-res1.min()
#     res1 = res1/res1.max()
#     res1[1]=0
#     res1[2]=0


    r2 = 0.5+0.5*r2/(np.absolute(r2).max())
    g2 = 0.5+0.5*g2/(np.absolute(g2).max())
    b2 = 0.5+0.5*b2/(np.absolute(b2).max())
    res2 = np.array([r2,g2,b2])
#     res2 = np.absolute(res2)
#     #print(res1.min())
#     res2 = res2-res2.min()
#     res2 = res2/res2.max()

    res1 = np.ascontiguousarray(res1.transpose(1, 2, 0))*255
    res2 = np.ascontiguousarray(res2.transpose(1, 2, 0))*255
    return res1,res2

def sep_SVD(conv1,conv2,dim=0):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
#     conv1 = (conv1.T/np.linalg.norm(conv1,axis=1)).T
#     conv2 = (conv2.T/np.linalg.norm(conv2,axis=1)).T
    conv1 = conv1 - conv1.mean(axis=0)
    conv2 = conv2 - conv2.mean(axis=0)
    u1, _, _ = np.linalg.svd(conv1, full_matrices=False)
    u2, _, _ = np.linalg.svd(conv2, full_matrices=False)
    
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))

#     r1 = np.reshape(u1[:,0],out_sz)
#     g1 = np.reshape(u1[:,1],out_sz)
#     b1 = np.reshape(u1[:,2],out_sz)
#     r2 = np.reshape(u2[:,0],out_sz)
#     g2 = np.reshape(u2[:,1],out_sz)
#     b2 = np.reshape(u2[:,2],out_sz)

#     res1 = np.array([r1,g1,b1])

#     res2 = np.array([r2,g2,b2])

#     res1 = np.ascontiguousarray(res1.transpose(1, 2, 0))
#     res2 = np.ascontiguousarray(res2.transpose(1, 2, 0))
#     res1 = res1/res1.max()
#     res2 = res2/res2.max()
#    return res1,res2
#     return np.absolute(np.reshape(u1[:,dim],out_sz)),np.absolute(np.reshape(u2[:,dim],out_sz))
    return np.reshape(u1[:,dim],out_sz),np.reshape(u2[:,dim],out_sz)
# def combine_SVD_test(conv1,conv2,weighs,feat1,feat2):
#     """
#     Takes in the last convolutional layer from two images, computes the pooled output
#     feature, and then generates the spatial similarity map for both images.
#     """
#     pool1 = np.mean(conv1,axis=0)
#     pool2 = np.mean(conv2,axis=0)
# #     conv1 = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
# #     conv2 = conv2 / np.linalg.norm(pool2) / conv2.shape[0]

# #     conv1 = (conv1.T-conv1.mean(axis=1)).T
# #     conv2 = (conv2.T-conv2.mean(axis=1)).T
    
#     allconv = np.vstack((conv1,conv2))
#     allconv = allconv-allconv.mean(axis=0)
#     u, _, _ = np.linalg.svd(allconv, full_matrices=False)
# #     u1, s, v= np.linalg.svd(conv1, full_matrices=False)
# #     u2, _, _ = np.linalg.svd(conv2, full_matrices=False)
#     out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
#     #newconv1_r = u1[0]*s[0]*v[0]
#     r1 = np.reshape(u[:,0][0:conv1.shape[0]],out_sz)
#     g1 = np.reshape(u[:,1][0:conv1.shape[0]],out_sz)
#     b1 = np.reshape(u[:,2][0:conv1.shape[0]],out_sz)
#     r2 = np.reshape(u[:,0][conv1.shape[0]:],out_sz)
#     g2 = np.reshape(u[:,1][conv1.shape[0]:],out_sz)
#     b2 = np.reshape(u[:,2][conv1.shape[0]:],out_sz)
# #     zero_matrix = np.zeros((8,8))
# #     r1 = np.reshape(u1[:,0],out_sz)
# #     g1 = np.reshape(u1[:,1],out_sz)
# #     b1 = np.reshape(u1[:,2],out_sz)
# #     r2 = np.reshape(u2[:,0],out_sz)
# #     g2 = np.reshape(u2[:,1],out_sz)
# #     b2 = np.reshape(u2[:,2],out_sz)

    
#     #res1 = np.array([r1,g1,b1])
# #     r1 = r1-r1.min()
# #     r1 = r1/r1.max()
# #     g1 = g1-g1.min()
# #     g1 = g1/g1.max()
# #     b1 = b1-b1.min()
# #     b1 = b1/b1.max()
#     r1 = 0.5+np.absolute(0.5*r1/(np.absolute(r1).max()))
#     g1 = 0.5+np.absolute(0.5*g1/(np.absolute(g1).max()))
#     b1 = 0.5+np.absolute(0.5*b1/(np.absolute(b1).max()))
#     res1 = np.array([r1,g1,b1])

# #     res1 = (res1-res1.min())
# #     res1 = res1/res1.max()
    
#     #res2 = np.array([r2,g2,b2])
# #     r2 = r2-r2.min()
# #     r2 = r2/r2.max()
# #     g2 = g2-g2.min()
# #     g2 = g2/g2.max()
# #     b2 = b2-b2.min()
# #     b2 = b2/b2.max()
# #     res2 = np.array([r2,g2,b2])
#     r2 = 0.5+np.absolute(0.5*r2/(np.absolute(r2).max()))
#     g2 = 0.5+np.absolute(0.5*g2/(np.absolute(g2).max()))
#     b2 = 0.5+np.absolute(0.5*b2/(np.absolute(b2).max()))
#     res2 = np.array([r2,g2,b2])

# #     res2 = (res2-res2.min())
# #     res2 = res2/res2.max()
#     res1 = np.ascontiguousarray(res1.transpose(1, 2, 0))*255
#     res2 = np.ascontiguousarray(res2.transpose(1, 2, 0))*255
#     return res1,res2


def vectors_norm(conv1,conv2,weighs,feat1,feat2):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    res1 = np.linalg.norm(conv1,axis=1)
    res2 = np.linalg.norm(conv2,axis=1)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    res1 = np.reshape(res1,out_sz)
    res2 = np.reshape(res2,out_sz)
    return res1,res2

def find_sim_distru(conv1,conv2,weighs,feat1,feat2):
#     feat1 = feat1/np.linalg.norm(feat1)
#     feat2 = feat2/np.linalg.norm(feat2)
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    conv1 = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    conv2 = conv2 / np.linalg.norm(pool2) / conv2.shape[0]    
    Mat_sim = np.dot(conv1,conv2.T)
    Msim = Mat_sim.copy()
    Mdif = Mat_sim.copy()
    sim = Mat_sim.mean()
    Msim[Msim<sim]=0
    Mdif[Mdif>sim]=0
    Mdif = -Mdif+sim
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    conv1_msim = np.amax(Mat_sim,axis=1)
    conv2_msim = np.amax(Mat_sim,axis=0)
    conv1_pos = np.reshape(np.amax(Msim,axis=1),out_sz)
    conv2_pos = np.reshape(np.amax(Msim,axis=0),out_sz)
    conv1_neg = np.reshape(np.amax(Mdif,axis=1),out_sz)
    conv2_neg = np.reshape(np.amax(Mdif,axis=0),out_sz)
    
    norm1 = np.linalg.norm(conv1,axis=1)
    norm2 = np.linalg.norm(conv2,axis=1)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    norm1 = np.reshape(norm1,out_sz)
    norm2 = np.reshape(norm2,out_sz)
    
    mask = conv1_msim==conv2_msim[::-1]
    res1 = conv1_msim.copy()
    res1[~mask]=0
    res2 = conv2_msim.copy()
    res2[~mask[::-1]]=0
    res1 = np.reshape(res1,out_sz)
    res2 = np.reshape(res2,out_sz)
    return norm1,norm2,res1,res2,Mat_sim.flatten(),conv1_pos,conv2_pos,conv1_neg,conv2_neg
    

def find_sim_distru_thres(conv1,conv2,weighs,feat1,feat2):
#     feat1 = feat1/np.linalg.norm(feat1)
#     feat2 = feat2/np.linalg.norm(feat2)
    thres = 0.1
    norm_thres = 0.5
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    conv1 = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    conv2 = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    
    norm1 = np.linalg.norm(conv1,axis=1)
    norm2 = np.linalg.norm(conv2,axis=1)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    norm1_res = np.reshape(norm1,out_sz)
    norm2_res = np.reshape(norm2,out_sz)
    
    norm1_thres = norm_thres*norm1.max()
    norm2_thres = norm_thres*norm2.max()
    
    conv1 = (conv1.T / norm1).T
    conv2 = (conv2.T / norm2).T
    
    for i in range(0,norm1.shape[0]):
        if norm1[i]<norm1_thres:
            conv1[i] = np.zeros(conv1.shape[1])
        if norm2[i]<norm2_thres:
            conv2[i] = np.zeros(conv2.shape[1])
    
    Mat_sim = np.dot(conv1,conv2.T)
    Mat_sim_flat = Mat_sim.flatten()
    Mat_sim_flat.sort()
    Mat_sim_flat = Mat_sim_flat[Mat_sim_flat!=0]
    
    thres = int(thres*Mat_sim_flat.shape[0])
    high_thres = Mat_sim_flat[-thres] 
    low_thres = Mat_sim_flat[thres]
    
    high_Mat = Mat_sim.copy()
    high_Mat[high_Mat<high_thres] = 0
    low_Mat = Mat_sim.copy()
    low_Mat_max = low_Mat.max()
    low_Mat[low_Mat>low_thres]=0
    low_Mat[low_Mat!=0]=high_Mat.max()
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    conv1_msim = np.amax(Mat_sim,axis=1)
    conv2_msim = np.amax(Mat_sim,axis=0)
    conv1_pos = np.reshape(np.amax(high_Mat,axis=1),out_sz)
    conv2_pos = np.reshape(np.amax(high_Mat,axis=0),out_sz)
    conv1_neg = np.reshape(np.amax(low_Mat,axis=1),out_sz)
    conv2_neg = np.reshape(np.amax(low_Mat,axis=0),out_sz)
    
    mask = conv1_msim==conv2_msim[::-1]
    res1 = conv1_msim.copy()
    res1[~mask]=0
    res2 = conv2_msim.copy()
    res2[~mask[::-1]]=0
    res1 = np.reshape(res1,out_sz)
    res2 = np.reshape(res2,out_sz)
    
    Dis_plot = sns.distplot(Mat_sim_flat)
    Dis_plot.figure.set_size_inches(2.56,2.56)
    Dis_plot.figure.canvas.draw()
    data = np.fromstring(Dis_plot.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(Dis_plot.figure.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return norm1_res,norm2_res,res1,res2,data,conv1_pos,conv2_pos,conv1_neg,conv2_neg

def find_sim_distru_thres_cor(conv1,conv2,weighs,feat1,feat2):
#     feat1 = feat1/np.linalg.norm(feat1)
#     feat2 = feat2/np.linalg.norm(feat2)
    thres = 0.1
    norm_thres = 0.5
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    conv1 = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    conv2 = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    
    norm1 = np.linalg.norm(conv1,axis=1)
    norm2 = np.linalg.norm(conv2,axis=1)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    norm1_res = np.reshape(norm1,out_sz)
    norm2_res = np.reshape(norm2,out_sz)
    
    norm1_thres = norm_thres*norm1.max()
    norm2_thres = norm_thres*norm2.max()
    
    conv1 = (conv1.T / norm1).T
    conv2 = (conv2.T / norm2).T
    
    for i in range(0,norm1.shape[0]):
        if norm1[i]<norm1_thres:
            conv1[i] = np.zeros(conv1.shape[1])
        if norm2[i]<norm2_thres:
            conv2[i] = np.zeros(conv2.shape[1])
    
    Mat_sim = np.dot(conv1,conv2.T)
    Mat_sim_flat = Mat_sim.flatten()
    Mat_sim_flat.sort()
    Mat_sim_flat = Mat_sim_flat[Mat_sim_flat!=0]
    
    thres = int(thres*Mat_sim_flat.shape[0])
    high_thres = Mat_sim_flat[-thres] 
    low_thres = Mat_sim_flat[thres]
    
    high_Mat = Mat_sim.copy()
    high_Mat[high_Mat<high_thres] = 0
    low_Mat = Mat_sim.copy()
    low_Mat_max = low_Mat.max()
    low_Mat[low_Mat>low_thres]=0
    low_Mat[low_Mat!=0]=high_Mat.max()
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    conv1_msim = np.amax(Mat_sim,axis=1)
    conv2_msim = np.amax(Mat_sim,axis=0)
    
    atob_pos = high_Mat.copy()
    for n,i in enumerate(atob_pos.T):
        atob_pos.T[n][atob_pos.T[n]!=i.max()]=0
    atob_pos = (atob_pos>0)*(np.arange(atob_pos.shape[0])).T
    
    atob_neg = low_Mat.copy()
    for n,i in enumerate(atob_neg.T):
        atob_neg.T[n][atob_neg.T[n]!=i.max()]=0
    atob_neg = (atob_neg>0)*(np.arange(atob_neg.shape[0])).T
    
    atob_conv1_pos = np.reshape(np.amax(atob_pos,axis=1),out_sz)
    atob_conv1_neg = np.reshape(np.amax(atob_neg,axis=1),out_sz)
    atob_conv2_pos = np.reshape(np.amax(atob_pos,axis=0),out_sz)
    atob_conv2_neg = np.reshape(np.amax(atob_neg,axis=0),out_sz)
    

    btoa_pos = high_Mat.copy()
    for n,i in enumerate(atob_pos):
        atob_pos[n][atob_pos[n]!=i.max()]=0
    btoa_pos = (btoa_pos>0)*np.arange(btoa_pos.shape[0])
    
    btoa_neg = low_Mat.copy()
    for n,i in enumerate(btoa_neg):
        btoa_neg[n][btoa_neg[n]!=i.max()]=0
    btoa_neg = (atob_neg>0)*np.arange(btoa_neg.shape[0])
    
    btoa_conv1_pos = np.reshape(np.amax(btoa_pos,axis=1),out_sz)
    btoa_conv1_neg = np.reshape(np.amax(btoa_neg,axis=1),out_sz)
    btoa_conv2_pos = np.reshape(np.amax(btoa_pos,axis=0),out_sz)
    btoa_conv2_neg = np.reshape(np.amax(btoa_neg,axis=0),out_sz)

    Dis_plot = sns.distplot(Mat_sim_flat)
    Dis_plot.figure.set_size_inches(2.56,2.56)
    Dis_plot.figure.canvas.draw()
    data = np.fromstring(Dis_plot.figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(Dis_plot.figure.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return norm1_res,norm2_res,data,atob_conv1_pos,atob_conv2_pos,atob_conv1_neg,atob_conv2_neg,btoa_conv1_pos,btoa_conv2_pos,btoa_conv1_neg,btoa_conv2_neg

def SVD_whole(convlist):

    allconv = np.zeros(convlist[0].shape)
    leng = len(convlist)
    for convmap in convlist:
#         pool = np.mean(convmap,axis=0)
        #convmap = convmap / np.linalg.norm(pool) / convmap.shape[0]
        allconv = np.vstack((allconv,convmap))

    allconv = allconv[convlist[0].shape[0]:,:]
    allconv =(allconv-allconv.mean(axis=0))
    print(allconv.mean(axis=0).shape)
    
    
    u, s, v = np.linalg.svd(allconv, full_matrices=False)
#     u1, s, v= np.linalg.svd(conv1, full_matrices=False)
#     u2, _, _ = np.linalg.svd(conv2, full_matrices=False)
    out_sz = (int(np.sqrt(convmap.shape[0])),int(np.sqrt(convmap.shape[0])))
    #newconv1_r = u1[0]*s[0]*v[0]
#     u = u*s
#     u = -u
    reslist = []
    for i in range(0,leng):
        r = np.reshape(u[:,0][convmap.shape[0]*i:convmap.shape[0]*(i+1)],out_sz)
        g = np.reshape(u[:,1][convmap.shape[0]*i:convmap.shape[0]*(i+1)],out_sz)
        b = np.reshape(u[:,2][convmap.shape[0]*i:convmap.shape[0]*(i+1)],out_sz)
#         r = np.absolute(r)
#         b = np.absolute(g)
#         g = np.absolute(b)
#         r = r-r.min()
#         r = r/r.max()
#         g = g-g.min()
#         g = g/g.max()
#         b = b-b.min()
#         b = b/b.max()
        r = 0.5+0.5*r/(np.absolute(r).max())
        g = 0.5+0.5*g/(np.absolute(g).max())
        b = 0.5+0.5*b/(np.absolute(b).max())
        res = np.array([r,g,b])
        #res = np.array([r,g,b])
        res = np.ascontiguousarray(res.transpose(1, 2, 0))*255
        reslist.append(res)

    return reslist

def SVD_whole_norm(convlist):

    allconv = np.zeros(convlist[0].shape)
    leng = len(convlist)
    for convmap in convlist:
#         pool = np.mean(convmap,axis=0)
        #convmap = convmap / np.linalg.norm(pool) / convmap.shape[0]
        allconv = np.vstack((allconv,convmap))

    allconv = allconv[convlist[0].shape[0]:,:]
    allconv =(allconv-allconv.mean(axis=0))
    print(allconv.mean(axis=0).shape)
    u, s, v = np.linalg.svd(allconv, full_matrices=False)
#     u1, s, v= np.linalg.svd(conv1, full_matrices=False)
#     u2, _, _ = np.linalg.svd(conv2, full_matrices=False)
    out_sz = (int(np.sqrt(convmap.shape[0])),int(np.sqrt(convmap.shape[0])))
    #newconv1_r = u1[0]*s[0]*v[0]
    u = u*s
#     u = -u
    reslist = []
    for i in range(0,leng):
        #res = np.reshape(np.linalg.norm(u[:,:64][convmap.shape[0]*i:convmap.shape[0]*(i+1)],axis=1),out_sz)
        res = np.reshape(np.linalg.norm(u[convmap.shape[0]*i:convmap.shape[0]*(i+1)],axis=1),out_sz)
        res = res.max()-res
        res = res/res.max()
        reslist.append(res)
    return reslist

def self_attention(conv):
    out_sz = (int(np.sqrt(conv.shape[0])),int(np.sqrt(conv.shape[0])))
    conv_mag = np.linalg.norm(conv,axis=1)
    conv = (conv.T/conv_mag).T
    selfdist = np.matmul(conv,conv.T)
    np.fill_diagonal(selfdist, 0)
    print(selfdist.shape)
    #m = scipy.special.softmax(selfdist, axis=1)
    m = selfdist.sum(axis=0) 
    m = np.reshape(m,out_sz)
    m = 1-m/m.max()
    return m

def SVD_pooling(convlist):

    allconv = np.zeros(convlist[0].shape)
    leng = len(convlist)
    for convmap in convlist:
#         pool = np.mean(convmap,axis=0)
        #convmap = convmap / np.linalg.norm(pool) / convmap.shape[0]
        allconv = np.vstack((allconv,convmap))

    allconv = allconv[convlist[0].shape[0]:,:]
    allconv =(allconv-allconv.mean(axis=0))
    print(allconv.mean(axis=0).shape)
    u, s, v = np.linalg.svd(allconv, full_matrices=False)
#     u1, s, v= np.linalg.svd(conv1, full_matrices=False)
#     u2, _, _ = np.linalg.svd(conv2, full_matrices=False)
    out_sz = (int(np.sqrt(convmap.shape[0])),int(np.sqrt(convmap.shape[0])))
    #newconv1_r = u1[0]*s[0]*v[0]
    u = u*s
#     u = -u
    reslist = []
    for i in range(0,leng):
        #res = np.reshape(np.linalg.norm(u[:,:64][convmap.shape[0]*i:convmap.shape[0]*(i+1)],axis=1),out_sz)
        res = np.reshape(np.linalg.norm(u[convmap.shape[0]*i:convmap.shape[0]*(i+1)],axis=1),out_sz)
        res = res/res.max()
        reslist.append(res)
    return reslist

def SVD_whole_norm(convlist):

    allconv = np.zeros(convlist[0].shape)
    leng = len(convlist)
    for convmap in convlist:
#         pool = np.mean(convmap,axis=0)
        #convmap = convmap / np.linalg.norm(pool) / convmap.shape[0]
        allconv = np.vstack((allconv,convmap))

    allconv = allconv[convlist[0].shape[0]:,:]
    allconv =(allconv-allconv.mean(axis=0))
    print(allconv.mean(axis=0).shape)
    u, s, v = np.linalg.svd(allconv, full_matrices=False)
#     u1, s, v= np.linalg.svd(conv1, full_matrices=False)
#     u2, _, _ = np.linalg.svd(conv2, full_matrices=False)
    out_sz = (int(np.sqrt(convmap.shape[0])),int(np.sqrt(convmap.shape[0])))
    #newconv1_r = u1[0]*s[0]*v[0]
    u = u*s
#     u = -u
    reslist = []
    for i in range(0,leng):
        #res = np.reshape(np.linalg.norm(u[:,:64][convmap.shape[0]*i:convmap.shape[0]*(i+1)],axis=1),out_sz)
        res = np.reshape(np.linalg.norm(u[convmap.shape[0]*i:convmap.shape[0]*(i+1)],axis=1),out_sz)
        res = res.max()-res
        res = res/res.max()
        reslist.append(res)
    return reslist

def SVD_whole_meanmax(convlist):
    meanreslist = []
    maxreslist = []
    for convmap in convlist:
        out_sz = (int(np.sqrt(convmap.shape[0])),int(np.sqrt(convmap.shape[0])))
        meanvect = convmap.mean(axis=0)
        meanvect = meanvect/np.linalg.norm(meanvect)
        maxvect = convmap.max(axis=0)
        maxvect = maxvect/np.linalg.norm(maxvect)
        print(meanvect.shape,maxvect.shape)
        
        conv_mag = np.linalg.norm(convmap,axis=1)
        convmap = (convmap.T/conv_mag).T
        
        meanres = np.matmul(convmap,meanvect)
        maxres = np.matmul(convmap,maxvect)
        meanres = np.reshape(meanres,out_sz)
        maxres = np.reshape(maxres,out_sz)
        
        meanres = meanres.max()-meanres
        meanres = meanres/meanres.max()
        
        maxres = maxres.max()-maxres
        maxres = maxres/maxres.max()
        
        meanreslist.append(meanres)
        maxreslist.append(maxres)
    return meanreslist,maxreslist

def SVD_whole_rgb(convlist):

    allconv = np.zeros(convlist[0].shape)
    leng = len(convlist)
    for convmap in convlist:
        pool = np.mean(convmap,axis=0)
        #convmap = convmap / np.linalg.norm(pool) / convmap.shape[0]
        convmap = (convmap.T-convmap.mean(axis=1)).T
        allconv = np.vstack((allconv,convmap))

    allconv = allconv[convlist[0].shape[0]:,:]
    print(allconv.mean(axis=0).shape)
    allconv =(allconv-allconv.mean(axis=0))
    u, _, _ = np.linalg.svd(allconv, full_matrices=False)
#     u1, s, v= np.linalg.svd(conv1, full_matrices=False)
#     u2, _, _ = np.linalg.svd(conv2, full_matrices=False)
    out_sz = (int(np.sqrt(convmap.shape[0])),int(np.sqrt(convmap.shape[0])))
    #newconv1_r = u1[0]*s[0]*v[0]
    
    reslist1 = []
    reslist2 = []
    reslist3 = []
    for i in range(0,leng):
        r = np.reshape(u[:,0][convmap.shape[0]*i:convmap.shape[0]*(i+1)],out_sz)
        g = np.reshape(u[:,1][convmap.shape[0]*i:convmap.shape[0]*(i+1)],out_sz)
        b = np.reshape(u[:,2][convmap.shape[0]*i:convmap.shape[0]*(i+1)],out_sz)
        r = np.absolute(r)
        g = np.absolute(g)
        b = np.absolute(b)
        reslist1.append(r)
        reslist2.append(g)
        reslist3.append(b)
    return reslist1,reslist2,reslist3


def SVD_vsmean(convlist):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    allconv = np.zeros(convlist[0].shape)
    leng = len(convlist)
    for convmap in convlist:
        pool = np.mean(convmap,axis=0)
        convmap = convmap / np.linalg.norm(pool) / convmap.shape[0]
        #convmap = (convmap.T-convmap.mean(axis=1)).T
        allconv = np.vstack((allconv,convmap))

    allconv = allconv[convlist[0].shape[0]:,:]
    convmean = allconv.mean(axis=0)
    out_sz = (int(np.sqrt(convmap.shape[0])),int(np.sqrt(convmap.shape[0])))
    reslist = []

    for i in range(0,leng):
        im_similarity = np.array([])
        for convvect in convlist[i]:
            cosine_distance = distance.cosine(convvect,convmean)
            im_similarity = np.append(im_similarity,cosine_distance)
        im_similarity = im_similarity.reshape(out_sz)    
        reslist.append(im_similarity)

    return reslist


def SVD_basis(convlist,n_dim=10):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    allconv = np.zeros(convlist[0].shape)
    leng = len(convlist)
#     for convmap in convlist:
#         pool = np.mean(convmap,axis=0)
#         convmap = convmap / np.linalg.norm(pool) / convmap.shape[0]
#         convmap = (convmap.T-convmap.mean(axis=1)).T
#         allconv = np.vstack((allconv,convmap))
#     allconv = allconv[convlist[0].shape[0]:,:]

    for convmap in convlist:
        pool = np.mean(convmap,axis=0)
#         convmap = convmap / np.linalg.norm(pool)
        allconv = np.vstack((allconv,convmap))
    allconv = allconv[convlist[0].shape[0]:,:]
    mean = allconv.mean(axis=0)
    allconv = allconv-mean
    u, s, v = np.linalg.svd(allconv, full_matrices=False)
#     u1, s, v= np.linalg.svd(conv1, full_matrices=False)
#     u2, _, _ = np.linalg.svd(conv2, full_matrices=False)
#     out_sz = (int(np.sqrt(convmap.shape[0])),int(np.sqrt(convmap.shape[0])))
#     #newconv1_r = u1[0]*s[0]*v[0]
    

    return v[0:n_dim]

def SVD_basis_gpu(convlist,n_dim=17,if_map=True):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    if if_map:
        allconv = np.zeros(convlist[0].shape)
        leng = len(convlist)
    #     for convmap in convlist:
    #         pool = np.mean(convmap,axis=0)
    #         convmap = convmap / np.linalg.norm(pool) / convmap.shape[0]
    #         convmap = (convmap.T-convmap.mean(axis=1)).T
    #         allconv = np.vstack((allconv,convmap))
    #     allconv = allconv[convlist[0].shape[0]:,:]

        for convmap in convlist:
            pool = np.mean(convmap,axis=0)
#             convmap = convmap / np.linalg.norm(pool)
            allconv = np.vstack((allconv,convmap))
        allconv = allconv[convlist[0].shape[0]:,:]
        allconv = allconv-allconv.mean(axis=0)
        allconv = torch.tensor(allconv).cuda()
        u,s,v = torch.svd_lowrank(allconv,n_dim)
    else:
        allconv = np.array(convlist)
        pool = np.mean(allconv,axis=0)
        allconv = allconv / np.linalg.norm(pool)
        allconv = torch.tensor(allconv).cuda()
        u,s,v = torch.svd_lowrank(allconv,n_dim)
#     u1, s, v= np.linalg.svd(conv1, full_matrices=False)
#     u2, _, _ = np.linalg.svd(conv2, full_matrices=False)
#     out_sz = (int(np.sqrt(convmap.shape[0])),int(np.sqrt(convmap.shape[0])))
#     #newconv1_r = u1[0]*s[0]*v[0]
    

    return v.T


def SVD_basis_fc(convlist,weighs,n_dim=3):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    allconv = np.zeros((convlist[0].shape[0],weighs.shape[0]))
    leng = len(convlist)
#     for convmap in convlist:
#         pool = np.mean(convmap,axis=0)
#         convmap = convmap / np.linalg.norm(pool) / convmap.shape[0]
#         convmap = (convmap.T-convmap.mean(axis=1)).T
#         allconv = np.vstack((allconv,convmap))
#     allconv = allconv[convlist[0].shape[0]:,:]

    for convmap in convlist:
        convmap = np.matmul(convmap, weighs.T)
        pool = np.mean(convmap,axis=0)
        convmap = convmap / np.linalg.norm(pool) / convmap.shape[0]
        allconv = np.vstack((allconv,convmap))
    allconv = allconv[convlist[0].shape[0]:,:]
    #allconv = allconv-allconv.mean(axis=0)
    u, _, v = np.linalg.svd(allconv, full_matrices=False)
#     u1, s, v= np.linalg.svd(conv1, full_matrices=False)
#     u2, _, _ = np.linalg.svd(conv2, full_matrices=False)
    out_sz = (int(np.sqrt(convmap.shape[0])),int(np.sqrt(convmap.shape[0])))
    #newconv1_r = u1[0]*s[0]*v[0]
    

    return v[0:n_dim]


def SVD_reconstruction(conv1,conv2,n_dim=196):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    pool1 = np.mean(conv1,axis=0)
    conv1 = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    pool2 = np.mean(conv2,axis=0)
    conv2 = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    u1, _, v1 = np.linalg.svd(conv1, full_matrices=False)
    u2, _, v2 = np.linalg.svd(conv2, full_matrices=False)
#     u1, s, v= np.linalg.svd(conv1, full_matrices=False)
#     u2, _, _ = np.linalg.svd(conv2, full_matrices=False)
#     out_sz = (int(np.sqrt(convmap.shape[0])),int(np.sqrt(convmap.shape[0])))
#     #newconv1_r = u1[0]*s[0]*v[0]
        
    temp_1,resid,rank,s = np.linalg.lstsq(v2[0:n_dim].T,conv1.T)
    proj_1 = np.matmul(v2[0:n_dim].T, temp_1)
    #result1 = np.linalg.norm(proj_1,axis=0)/np.linalg.norm(conv1,axis=1)
    
    temp_1_self,resid,rank,s = np.linalg.lstsq(v1[0:n_dim].T,conv1.T)
    proj_1_self = np.matmul(v1[0:n_dim].T, temp_1_self)
    #result1_self = np.linalg.norm(proj_1_self,axis=0)/np.linalg.norm(conv1,axis=1)
    
    #result1 = np.divide(result1,result1_self)
    result1 = np.linalg.norm(proj_1,axis=0)/np.linalg.norm(proj_1_self,axis=0)
    #result1 = np.linalg.norm(proj_1,axis=0)/np.linalg.norm(conv1,axis=1)
    result1[np.isinf(result1)]=0
    result1[np.isnan(result1)]=0

    
    
    result1 = result1-result1.min()
    result1 = result1/result1.max()
    result1 = 1-result1
    result1 = result1.reshape(out_sz)
    
    temp_2,resid,rank,s = np.linalg.lstsq(v1[0:n_dim].T,conv2.T)
    proj_2 = np.matmul(v1[0:n_dim].T, temp_2)
    #result2 = np.linalg.norm(proj_2,axis=0)/np.linalg.norm(conv2,axis=1)
    
    temp_2_self,resid,rank,s = np.linalg.lstsq(v2[0:n_dim].T,conv2.T)
    proj_2_self = np.matmul(v2[0:n_dim].T, temp_2_self)
    #result2_self = np.linalg.norm(proj_2_self,axis=0)/np.linalg.norm(conv2,axis=1)
    
    #result2 = np.divide(result2,result2_self)
    result2 = np.linalg.norm(proj_2,axis=0)/np.linalg.norm(proj_2_self,axis=0)
    #result2 = np.linalg.norm(proj_2_self,axis=0)/np.linalg.norm(conv2,axis=1)
    result2[np.isinf(result2)]=0
    result2[np.isnan(result2)]=0

    
    result2 = result2-result2.min()
    result2 = result2/result2.max()
    result2 = 1-result2
    result2 = result2.reshape(out_sz)
    

    return result1,result2


def SVD_reconstruction_thres(conv1,conv2,n_dim=3):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    mean1 = conv1.mean(axis=0)
    mean2 = conv2.mean(axis=0)
#     conv1 = conv1-conv1.mean(axis=0)
#     conv2 = conv2-conv2.mean(axis=0)
#     pool1 = np.mean(conv1,axis=0)
#     conv1 = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
#     pool2 = np.mean(conv2,axis=0)
#     conv2 = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    
    
#     conv1_mag = np.linalg.norm(conv1,axis=1)
#     conv2_mag = np.linalg.norm(conv2,axis=1)
    
#     conv1 = (conv1.T/conv1_mag).T
#     conv2 = (conv1.T/conv2_mag).T
    
    u1, _, v1 = np.linalg.svd(conv1-mean1, full_matrices=False)
    u2, _, v2 = np.linalg.svd(conv2-mean2, full_matrices=False)
#     u1, s, v= np.linalg.svd(conv1, full_matrices=False)
#     u2, _, _ = np.linalg.svd(conv2, full_matrices=False)
#     out_sz = (int(np.sqrt(convmap.shape[0])),int(np.sqrt(convmap.shape[0])))
#     #newconv1_r = u1[0]*s[0]*v[0]
        
    temp_1,resid,rank,s = np.linalg.lstsq(v2[0:n_dim].T,(conv1-mean2).T)
    proj_1 = np.matmul(v2[0:n_dim].T, temp_1)
    
    #result1 = np.linalg.norm(proj_1,axis=0)/np.linalg.norm(conv1,axis=1)
    
        
    proj_1 = (proj_1/np.linalg.norm(proj_1,axis=0)).T
    conv1 = (conv1-mean2).T/np.linalg.norm((conv1-mean2),axis=1)
    result1 = np.einsum('ij,ji->i',proj_1,conv1)
    
    #temp_1_self,resid,rank,s = np.linalg.lstsq(v1[0:n_dim].T,conv1.T)
    #proj_1_self = np.matmul(v1[0:n_dim].T, temp_1_self)
    #result1_self = np.linalg.norm(proj_1_self,axis=0)/np.linalg.norm(conv1,axis=1)
    
    #result1 = np.divide(result1,result1_self)
    #result1 = np.linalg.norm(proj_1,axis=0)/np.linalg.norm(proj_1_self,axis=0)
    #result1 = np.linalg.norm(proj_1,axis=0)/np.linalg.norm(conv1,axis=1)
    result1[np.isinf(result1)]=0
    result1[np.isnan(result1)]=0

    
    #result1[conv1_mag<0.5*conv1_mag.mean()]=result1.min()
#     result1 = result1-result1.min()
#     result1 = result1/result1.max()
    result1 = 1-result1
    result1 = result1.reshape(out_sz)
    
    temp_2,resid,rank,s = np.linalg.lstsq(v1[0:n_dim].T,(conv2-mean1).T)
    proj_2 = np.matmul(v1[0:n_dim].T, temp_2)
    
    #result2 = np.linalg.norm(proj_2,axis=0)/np.linalg.norm(conv2,axis=1)
    
    proj_2 = (proj_2/np.linalg.norm(proj_2,axis=0)).T
    conv2 = (conv2-mean1).T/np.linalg.norm((conv2-mean1),axis=1)
    result2 = np.einsum('ij,ji->i',proj_2,conv2)    
    
    #temp_2_self,resid,rank,s = np.linalg.lstsq(v2[0:n_dim].T,conv2.T)
    #proj_2_self = np.matmul(v2[0:n_dim].T, temp_2_self)
    #result2_self = np.linalg.norm(proj_2_self,axis=0)/np.linalg.norm(conv2,axis=1)
    
    #result2 = np.divide(result2,result2_self)
    #result2 = np.linalg.norm(proj_2,axis=0)/np.linalg.norm(proj_2_self,axis=0)
    #result2 = np.linalg.norm(proj_2_self,axis=0)/np.linalg.norm(conv2,axis=1)
    result2[np.isinf(result2)]=0
    result2[np.isnan(result2)]=0
    #result2[conv2_mag<0.5*conv2_mag.mean()]=result2.min()
#     result2 = result2-result2.min()
#     result2 = result2/result2.max()
    result2 = 1-result2
    result2 = result2.reshape(out_sz)
    return result1,result2

def one_recons(conv,basis, other_mean = 0,n_dim=3):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    out_sz = (int(np.sqrt(conv.shape[0])),int(np.sqrt(conv.shape[0])))   
    conv = conv-other_mean
    temp_1,resid,rank,s = np.linalg.lstsq(basis.T,conv.T)
    proj_1 = np.matmul(basis.T, temp_1)
    
    #result1 = np.linalg.norm(proj_1,axis=0)/np.linalg.norm(conv1,axis=1)
    
    
    proj_1 = (proj_1/np.linalg.norm(proj_1,axis=0)).T
    conv = conv.T/np.linalg.norm(conv,axis=1)
    result1 = np.einsum('ij,ji->i',proj_1,conv)
    
    #temp_1_self,resid,rank,s = np.linalg.lstsq(v1[0:n_dim].T,conv1.T)
    #proj_1_self = np.matmul(v1[0:n_dim].T, temp_1_self)
    #result1_self = np.linalg.norm(proj_1_self,axis=0)/np.linalg.norm(conv1,axis=1)
    
    #result1 = np.divide(result1,result1_self)
    #result1 = np.linalg.norm(proj_1,axis=0)/np.linalg.norm(proj_1_self,axis=0)
    #result1 = np.linalg.norm(proj_1,axis=0)/np.linalg.norm(conv1,axis=1)
    result1[np.isinf(result1)]=0
    result1[np.isnan(result1)]=0

    
    #result1[conv1_mag<0.5*conv1_mag.mean()]=result1.min()
#     result1 = result1-result1.min()
#     result1 = result1/result1.max()
    result1 = 1-result1
    result1 = result1.reshape(out_sz)
    return result1

def one_recons_rgb(conv,basis,n_dim=3):
    """
    Takes in the last convolutional layer from two images, computes the pooled output
    feature, and then generates the spatial similarity map for both images.
    """
    out_sz = (int(np.sqrt(conv.shape[0])),int(np.sqrt(conv.shape[0])))        
    temp_1,resid,rank,s = np.linalg.lstsq(basis.T,conv.T)
    r = np.reshape(temp_1[0],out_sz)
    g = np.reshape(temp_1[1],out_sz)
    b = np.reshape(temp_1[2],out_sz)
    r = 0.5+0.5*r/(np.absolute(r).max())
    g = 0.5+0.5*g/(np.absolute(g).max())
    b = 0.5+0.5*b/(np.absolute(b).max())
    res = np.array([r,g,b])

    #res = np.array([r,g,b])
    res = np.ascontiguousarray(res.transpose(1, 2, 0))*255
    return res