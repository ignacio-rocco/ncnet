import torch
import torch.nn
import numpy as np
import os
from skimage import draw
import torch.nn.functional as F
from torch.autograd import Variable
from lib.pf_dataset import PFPascalDataset
from lib.point_tnf import PointsToUnitCoords, PointsToPixelCoords, bilinearInterpPointTnf


def pck(source_points,warped_points,L_pck,alpha=0.1):
    # compute precentage of correct keypoints
    batch_size=source_points.size(0)
    pck=torch.zeros((batch_size))
    for i in range(batch_size):
        p_src = source_points[i,:]
        p_wrp = warped_points[i,:]
        N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
        L_pck_mat = L_pck[i].expand_as(point_distance)
        correct_points = torch.le(point_distance,L_pck_mat*alpha)
        pck[i]=torch.mean(correct_points.float())
    return pck


def pck_metric(batch,batch_start_idx,matches,stats,args,use_cuda=True):
       
    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_points = batch['source_points']
    target_points = batch['target_points']
    
    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(target_points,target_im_size)

    # compute points stage 1 only
    warped_points_norm = bilinearInterpPointTnf(matches,target_points_norm)
    warped_points = PointsToPixelCoords(warped_points_norm,source_im_size)
    
    L_pck = batch['L_pck'].data
    
    current_batch_size=batch['source_im_size'].size(0)
    indices = range(batch_start_idx,batch_start_idx+current_batch_size)

    # compute PCK
    pck_batch = pck(source_points.data, warped_points.data, L_pck)
    stats['point_tnf']['pck'][indices] = pck_batch.unsqueeze(1).cpu().numpy()
        
    return stats