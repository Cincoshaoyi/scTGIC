# scTGIC

Deep Information Fusion Based on a Transformer Graph Encoder for a Single-Cell Multiomics Clustering

### Requirement
Python = 3.8.20
torch = 2.0.0
numpy = 1.24.3
munkres = 1.1.4
sklearn = 1.3.0
munkres = 1.1.4
scipy  = 1.10.1


### Usages

- Take the dataset "inhouse" as an example

- Pre-training
> python main.py --name inhouse --pretrain True
- training
> python main.py --name inhouse

##  Comparison methods availability

| Method |link | 
|----------------|--------------------------------| 
|DCCA|https://github.com/cmzuo11/DCCA|
|SMILE|https://github.com/rpmccordlab/SMILE |
|scMIC |https://github.com/Zyl-SZU/scMIC|
|sc-spectrum |https://github.com/jssong-lab/sc-spectrum|
|MoClust |https://zenodo.org/records/7306504|

## Data availability
| Dataset |link | 
|----------------|--------------------------------| 
|Inhouse|https://github.com/LongLVv/DEMOC_code/tree/main/data/In_house_PBMC2000|
|CellMix|https://github.com/cmzuo11/DCCA/blob/main/Example_test/scRNA_seq_SNARE.tsv |
|PBMC3k |https://zenodo.org/records/4762065|
|PBMCCITE |https://github.com/tarot0410/BREMSC/tree/master/data/RealData/10X10k|
|PBMC10k |https://scglue.readthedocs.io/en/latest/data.html|
