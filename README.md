# Persistent Hochschild Homology of Path Algebra

This is a project dedicated to computation of persistent Hochschild homology of directed graph. <br />
The method is described in the [article](https://arxiv.org/abs/2204.00462).
n-path digraph is used as a connectivity digraph.


## Running code
You need to prepare a configuration file in `.yaml` format with three parameters:
- `n` --- the dimensionality of simplices in the n-path digraph.
- `adj_mats_dir` --- the directory with adjacency matrices stored in `.npy` format.
- `res_dir` --- the directory for results.
  
Then execute`python ./comp_hh.py ./configs/config.yml`
