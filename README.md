# Neighbourhood Consensus Networks

![](https://www.di.ens.fr/willow/research/ncnet/images/teaser.png)


## About

This is the implementation of the paper "Neighbourhood Consensus Networks" by I. Rocco, M. Cimpoi, R. Arandjelović, A. Torii, T. Pajdla and J. Sivic.

For more information check out the project [[website](http://www.di.ens.fr/willow/research/ncnet/)] and the paper on [[arXiv](https://arxiv.org/abs/1810.10510)].


## Getting started

### Dependencies

The code is implemented using Python 3 and PyTorch 0.3. All dependencies should be included in the standard Anaconda distribution.

### Getting the datasets

The PF-Pascal dataset can be downloaded and unzipped by browsing to the `datasets/pf-pascal/` folder and running `download.sh`.

The IVD dataset (used for training for the InLoc benchmark) can be downloaded by browsing to the `datasets/ivd/` folder and first running `make_dirs.sh` and then `download.sh`.

The InLoc dataset (used for evaluation) an be downloaded by browsing to the `datasets/inloc/` folder and running `download.sh`. 

### Getting the trained models

The trained models trained on PF-Pascal (`ncnet_pfpascal.pth.tar`) and IVD (`ncnet_ivd.pth.tar`) can be dowloaded by browsing to the `trained_models/` folder and running `download.sh`.

### Keypoint transfer demo

The demo Jupyter notebook file `point_transfer_demo.py` illustrates how to evaluate the model and use it for keypoint transfer on the PF-Pascal dataset. For this, previously download the PF-Pascal dataset and trained model as indicated above.

## Training

To train a model, run `train.py` with the desired model architecture and the path to the training dataset.

Eg. For PF-Pascal:

```bash
python train.py --ncons_kernel_sizes 5 5 5 --ncons_channels 16 16 1 --dataset_image_path datasets/pf-pascal --dataset_csv_path datasets/pf-pascal/image_pairs/ 
```

Eg. For InLoc: 

```bash
python train.py --ncons_kernel_sizes 3 3 --ncons_channels 16 1 --dataset_image_path datasets/ivd --dataset_csv_path datasets/ivd/image_pairs/ 
```

## Evaluation

Evaluation for PF-Pascal is implemented in the `eval_pf_pascal.py` file. You can run the evaluation in the following way: 

```bash
python eval_pf_pascal.py --checkpoint trained_models/[checkpoint name]
```

Evaluation for InLoc is implemented in the `eval_inloc.py` file. You can run the evaluation in the following way: 

```bash
python eval_inloc.py --checkpoint trained_models/[checkpoint name]
```

This will generate a series of matches files in the `matches/` folder that then need to be fed to the InLoc evaluation Matlab code. 
In order to run the Matlab evaluation, you first need to clone the [InLoc demo repo](https://github.com/HajimeTaira/InLoc_demo), and download and compile all the required depedencies. Then you can modify the `compute_densePE_NCNet.m` file provided in this repo to indicate the path of the InLoc demo repo, and the name of the experiment (the particular folder name inside `matches/`), and run it to perform the evaluation.


## BibTeX 

If you use this code in your project, please cite our paper:
````
@InProceedings{Rocco18b,
        author       = "Rocco, I. and Cimpoi, M. and Arandjelovi\'c, R. and Torii, A. and Pajdla, T. and Sivic, J."
        title        = "Neighbourhood Consensus Networks",
        booktitle    = "Proceedings of the 32nd Conference on Neural Information Processing Systems",
        year         = "2018",
        }
````


