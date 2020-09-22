# README #

Source code associated to the paper
## ALPINE: Active Link Prediction Using Network Embedding (Under review)

### Requirements:
* Python 3.6+
* numpy 1.15.3
* scipy 1.1.0
* pandas 0.23.4
* matplotlib 3.0.0
* Conditional Network Embedding (modified from https://bitbucket.org/ghentdatascience/cne)


### Datasets
* The Harry Potter network: https://github.com/efekarakus/potter-network
* Polbooks: http://www-personal.umich.edu/~mejn/netdata/
* C.elegans: http://www-personal.umich.edu/~mejn/netdata/
* USAir http://vlado.fmf.uni-lj.si/pub/networks/data/
* MP_cc: We collect the data in April 2019 for the Members of Parliament in the UK. It originally has 650 nodes and we only use the largest connected component of 567 nodes and 49631 friendships.
* Polblogs_cc: http://www-personal.umich.edu/~mejn/netdata/ We use its largest connected component.
* PPI_cc: https://snap.stanford.edu/node2vec/ We use its largest connected component.
* Blog: https://snap.stanford.edu/node2vec/

### Run
1. Select and load a dataset in run.py.
2. Choose Case (i.e., 1, 2, 3), set the values of r_0 (the initially observed portion of node pairs), nr_split and nr_ne (the averaging parameters that can be set small in order to save time).
3. 'python run.py'

*Note that for large networks, e.g., dataset blog, it takes large memory and a few hours to iterate 5 times. More specifically, to run experiments on blog network, parallel computation for all the strategies might cause memory error if the device memory is not enough. But you can still run it sequentially. An example for the run time - blog with r_0 being 10% for Case-2 would take approximately 4 hours. It is also possible to define your own PON, as well as the pool and the target set.*


### Results
The results are visualized in the 'folder' defined for the this experiment named 'results.png'. See line 129~145 in run.py: the folder with results for different cases start with:

* Case-1 - 'TU_PU_r0...'
* Case-2 - 'TU_r0...'
* Case-3 - 'r0_...'
