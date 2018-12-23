from __future__ import print_function, division
import os
from os.path import exists, join, basename
from collections import OrderedDict
import numpy as np
import numpy.random
import scipy as sc
import scipy.misc
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable

from lib.model import ImMatchNet,MutualMatching
from lib.normalization import NormalizeImageDict
from lib.torch_util import str_to_bool
from lib.point_tnf import normalize_axis,unnormalize_axis,corr_to_matches
from lib.plot import plot_image

import argparse

print('NCNet evaluation script - InLoc dataset')

use_cuda = torch.cuda.is_available()

# Argument parsing
parser = argparse.ArgumentParser(description='Compute InLoc matches')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--inloc_shortlist', type=str, default='datasets/inloc/densePE_top100_shortlist_cvpr18.mat')
parser.add_argument('--k_size', type=int, default=2)
parser.add_argument('--image_size', type=int, default=3200)
parser.add_argument('--n_queries', type=int, default=356)
parser.add_argument('--n_panos', type=int, default=10)
parser.add_argument('--softmax', type=str_to_bool, default=True)
parser.add_argument('--matching_both_directions', type=str_to_bool, default=True)
parser.add_argument('--flip_matching_direction', type=str_to_bool, default=False)
parser.add_argument('--pano_path', type=str, default='datasets/inloc/pano/', help='path to InLoc panos - should contain CSE3,CSE4,CSE5,DUC1 and DUC2 folders')
parser.add_argument('--query_path', type=str, default='datasets/inloc/query/iphone7/', help='path to InLoc queries')

args = parser.parse_args()

image_size = args.image_size
k_size = args.k_size
matching_both_directions = args.matching_both_directions
flip_matching_direction = args.flip_matching_direction

# Load pretrained model
half_precision=True # use for memory saving

print(args)
    
model = ImMatchNet(use_cuda=use_cuda,
                   checkpoint=args.checkpoint,
                   half_precision=half_precision,
                   relocalization_k_size=args.k_size)

# Generate output folder path
output_folder = args.inloc_shortlist.split('/')[-1].split('.')[0]+'_SZ_NEW_'+str(image_size)+'_K_'+str(k_size)
if matching_both_directions:
    output_folder += '_BOTHDIRS'
elif flip_matching_direction:
    output_folder += '_AtoB'
else:
    output_folder += '_BtoA'
if args.softmax==True:
    output_folder += '_SOFTMAX'
if args.checkpoint!='':
    checkpoint_name=args.checkpoint.split('/')[-1].split('.')[0]
    output_folder += '_CHECKPOINT_'+checkpoint_name
print('Output matches folder: '+output_folder)

# Data preprocessing

# Manually change image resolution for this test. On training, image_size=400 was used, with squared images
scale_factor = 0.0625

imreadth = lambda x: torch.Tensor(sc.misc.imread(x).astype(np.float32)).transpose(1,2).transpose(0,1)
normalize = lambda x: NormalizeImageDict(['im'])({'im':x})['im']

# allow rectangular images. Does not modify aspect ratio.
if k_size==1:
    resize = lambda x: nn.functional.upsample(Variable(x.unsqueeze(0).cuda(),volatile=True),
                size=(int(x.shape[1]/(np.max(x.shape[1:])/image_size)),int(x.shape[2]/(np.max(x.shape[1:])/image_size))),mode='bilinear')
else:
    resize = lambda x: nn.functional.upsample(Variable(x.unsqueeze(0).cuda(),volatile=True),
                size=(int(np.floor(x.shape[1]/(np.max(x.shape[1:])/image_size)*scale_factor/k_size)/scale_factor*k_size),
                      int(np.floor(x.shape[2]/(np.max(x.shape[1:])/image_size)*scale_factor/k_size)/scale_factor*k_size)),mode='bilinear')

padim = lambda x,h_max: torch.cat((x,x.view(-1)[0].clone().expand(1,3,h_max-x.shape[2],x.shape[3])/1e20),dim=2) if x.shape[2]<h_max else x


# Get shortlists for each query image
shortlist_fn = args.inloc_shortlist

dbmat = loadmat(shortlist_fn)
db = dbmat['ImgList'][0,:]

query_fn_all=np.squeeze(np.vstack(tuple([db[q][0] for q in range(len(db))])))
pano_fn_all=np.vstack(tuple([db[q][1] for q in range(len(db))]))

Nqueries=args.n_queries
Npanos=args.n_panos

try:
    os.mkdir('matches/')
except FileExistsError:
    pass

try:
    os.mkdir('matches/'+output_folder)
except FileExistsError:
    pass

N=int((image_size*scale_factor/k_size)*np.floor((image_size*scale_factor/k_size)*(3/4)))
if matching_both_directions:
    N=2*N
    
do_softmax = args.softmax

plot=False

for q in range(Nqueries): 
    print(q)
    matches=numpy.zeros((1,Npanos,N,5))
    # load query image
    query_fn = os.path.join(args.query_path,db[q][0].item())
    src=resize(normalize(imreadth(query_fn)))
    
    # load database image
    for idx in range(Npanos):
        pano_fn = os.path.join(args.pano_path,db[q][1].ravel()[idx].item())
        tgt=resize(normalize(imreadth(pano_fn)))

        if k_size>1:
            corr4d,delta4d=model({'source_image':src,'target_image':tgt})
        else:
            corr4d=model({'source_image':src,'target_image':tgt})
            delta4d=None
            
        # reshape corr tensor and get matches for each point in image B
        batch_size,ch,fs1,fs2,fs3,fs4 = corr4d.size()
      
        # pad image and plot
        if plot:
            h_max=int(np.max([src.shape[2],tgt.shape[2]]))
            im=plot_image(torch.cat((padim(src,h_max),padim(tgt,h_max)),dim=3),return_im=True)
            plt.imshow(im)
    
        if matching_both_directions:
            (xA_,yA_,xB_,yB_,score_)=corr_to_matches(corr4d,scale='positive',do_softmax=do_softmax,delta4d=delta4d,k_size=k_size)
            (xA2_,yA2_,xB2_,yB2_,score2_)=corr_to_matches(corr4d,scale='positive',do_softmax=do_softmax,delta4d=delta4d,k_size=k_size,invert_matching_direction=True)
            xA_=torch.cat((xA_,xA2_),1)
            yA_=torch.cat((yA_,yA2_),1)
            xB_=torch.cat((xB_,xB2_),1)
            yB_=torch.cat((yB_,yB2_),1)
            score_=torch.cat((score_,score2_),1)
            # sort in descending score (this will keep the max-score instance in the duplicate removal step)
            sorted_index=torch.sort(-score_)[1].squeeze()
            xA_=xA_.squeeze()[sorted_index].unsqueeze(0)
            yA_=yA_.squeeze()[sorted_index].unsqueeze(0)
            xB_=xB_.squeeze()[sorted_index].unsqueeze(0)
            yB_=yB_.squeeze()[sorted_index].unsqueeze(0)
            score_=score_.squeeze()[sorted_index].unsqueeze(0)
            # remove duplicates
            concat_coords=np.concatenate((xA_.cpu().data.numpy(),yA_.cpu().data.numpy(),xB_.cpu().data.numpy(),yB_.cpu().data.numpy()),0)
            _,unique_index=np.unique(concat_coords,axis=1,return_index=True)
            xA_=xA_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
            yA_=yA_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
            xB_=xB_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
            yB_=yB_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
            score_=score_.squeeze()[torch.cuda.LongTensor(unique_index)].unsqueeze(0)
        elif flip_matching_direction:
            (xA_,yA_,xB_,yB_,score_)=corr_to_matches(corr4d,scale='positive',do_softmax=do_softmax,delta4d=delta4d,k_size=k_size,invert_matching_direction=True)
        else:
            (xA_,yA_,xB_,yB_,score_)=corr_to_matches(corr4d,scale='positive',do_softmax=do_softmax,delta4d=delta4d,k_size=k_size)
            
        # recenter
        if k_size>1:
            yA_=yA_*(fs1*k_size-1)/(fs1*k_size)+0.5/(fs1*k_size)
            xA_=xA_*(fs2*k_size-1)/(fs2*k_size)+0.5/(fs2*k_size)
            yB_=yB_*(fs3*k_size-1)/(fs3*k_size)+0.5/(fs3*k_size)
            xB_=xB_*(fs4*k_size-1)/(fs4*k_size)+0.5/(fs4*k_size)    
        else:
            yA_=yA_*(fs1-1)/fs1+0.5/fs1
            xA_=xA_*(fs2-1)/fs2+0.5/fs2
            yB_=yB_*(fs3-1)/fs3+0.5/fs3
            xB_=xB_*(fs4-1)/fs4+0.5/fs4
        
        xA = xA_.view(-1).data.cpu().float().numpy()
        yA = yA_.view(-1).data.cpu().float().numpy()
        xB = xB_.view(-1).data.cpu().float().numpy()
        yB = yB_.view(-1).data.cpu().float().numpy()
        score = score_.view(-1).data.cpu().float().numpy()
        
        Npts=len(xA)
        if Npts>0:
            matches[0,idx,:Npts,0]=xA
            matches[0,idx,:Npts,1]=yA
            matches[0,idx,:Npts,2]=xB
            matches[0,idx,:Npts,3]=yB
            matches[0,idx,:Npts,4]=score
        
            # plot top N matches
            if plot:
                c=numpy.random.rand(Npts,3)
                for i in range(Npts):       
                    if score[i]>0.75:
                        ax = plt.gca()
                        ax.add_artist(plt.Circle((float(xA[i])*src.shape[3],float(yA[i])*src.shape[2]), radius=3, color=c[i,:]))
                        ax.add_artist(plt.Circle((float(xB[i])*tgt.shape[3]+src.shape[3] ,float(yB[i])*tgt.shape[2]), radius=3, color=c[i,:]))
                        #plt.plot([float(xA[i])*src.shape[3], float(xB[i])*tgt.shape[3]+src.shape[3]], [float(yA[i])*src.shape[2], float(yB[i])*tgt.shape[2]], c='g', linestyle='-', linewidth=0.1)

        corr4d=None
        delta4d=None
        
        if idx%10==0:
            print(">>>"+str(idx))
            
    savemat(os.path.join('matches/',output_folder,str(q+1)+'.mat'),{'matches':matches,'query_fn':db[q][0].item(),'pano_fn':pano_fn_all},do_compression=True)
    
if plot:
    plt.gcf().set_dpi(200)
    plt.show()

