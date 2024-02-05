# Medium Biosciences - ML Hands-On Challenge

Welcome to the companion repository for the Medium Biosciences ML Hands-On Challenge. It has companion code and files for linking protein sequence, structure and their CATH architecture labels.

#### Files and Their Descriptions:

- `cath_w_seqs_share.csv` : It contains information protein sequences and hierarchies in the CATH classification
- `getting_started.ipynb` : Notebook for visualizing protein structures and classification
- `pdb_share.zip` : Zip files containing protein 3D structures.  
    Note: this file was uploaded using github lfs, so you may need to install this to download and access it.

## Setup

- In my case, I had to install Git LFS with `brew install git-lfs`. To get the input dataset, I ran `git lfs fetch <FILENAME>` in the original repository and then copied the file here. The commands for Git LFS are [here](https://www.mankier.com/1/git-lfs)
- Pytorch supports up to Python 3.11, so I set up this `Pipenv` environment with `pipenv install --python 3.11`.
- To get Pytorch: `pip3 install torch torchvision`