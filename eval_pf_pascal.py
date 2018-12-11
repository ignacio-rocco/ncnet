from __future__ import print_function, division
import os
from os.path import exists
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

from lib.model import ImMatchNet
from lib.pf_dataset import PFPascalDataset
from lib.normalization import NormalizeImageDict
from lib.torch_util import BatchTensorToVars, str_to_bool
from lib.point_tnf import corr_to_matches
from lib.eval_util import pck_metric
from lib.dataloader import default_collate
from lib.torch_util import collate_custom

import argparse


print('NCNet evaluation script - PF Pascal dataset')

use_cuda = torch.cuda.is_available()

# Argument parsing
parser = argparse.ArgumentParser(description='Compute PF Pascal matches')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--image_size', type=int, default=400)
parser.add_argument('--eval_dataset_path', type=str, default='datasets/pf-pascal/', help='path to PF Pascal dataset')

args = parser.parse_args()

# Create model
print('Creating CNN model...')
model = ImMatchNet(use_cuda=use_cuda,
                   checkpoint=args.checkpoint)

# Dataset and dataloader
Dataset = PFPascalDataset
collate_fn = default_collate
csv_file = 'image_pairs/test_pairs.csv'    
    
cnn_image_size=(args.image_size,args.image_size)

dataset = Dataset(csv_file=os.path.join(args.eval_dataset_path, csv_file),
                  dataset_path=args.eval_dataset_path,
                  transform=NormalizeImageDict(['source_image','target_image']),
                  output_size=cnn_image_size)
dataset.pck_procedure='scnet'

# Only batch_size=1 is supported for evaluation
batch_size=1

dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0,
                        collate_fn=collate_fn)

batch_tnf = BatchTensorToVars(use_cuda=use_cuda)
   
model.eval()    

# initialize vector for storing results
stats={}       
stats['point_tnf']={}
stats['point_tnf']['pck'] = np.zeros((len(dataset),1))

# Compute
for i, batch in enumerate(dataloader):
    batch = batch_tnf(batch)        
    batch_start_idx=batch_size*i

    corr4d = model(batch)

    # get matches
    xA,yA,xB,yB,sB=corr_to_matches(corr4d,do_softmax=True)
        
    matches=(xA,yA,xB,yB)
    stats = pck_metric(batch,batch_start_idx,matches,stats,args,use_cuda)
        
    print('Batch: [{}/{} ({:.0f}%)]'.format(i, len(dataloader), 100. * i / len(dataloader)))

# Print results
results=stats['point_tnf']['pck']
good_idx = np.flatnonzero((results!=-1) * ~np.isnan(results))
print('Total: '+str(results.size))
print('Valid: '+str(good_idx.size)) 
filtered_results = results[good_idx]
print('PCK:','{:.2%}'.format(np.mean(filtered_results)))