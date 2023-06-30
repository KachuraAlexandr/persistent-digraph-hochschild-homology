import numpy as np
from tqdm import tqdm
import os
import sys
import yaml

from dfl import DirectedFlagComplex
        
        
def main():
    config_fname = sys.argv[1]
    with open(config_fname, 'r') as config_f:
        config = yaml.load(config_f, Loader=yaml.Loader)
    
    n = config['n']
    adj_mats_dir = config['adj_mats_dir']
    res_dir = config['res_dir']
    
    for i, adj_mat_fname in tqdm(enumerate(os.listdir(adj_mats_dir))):
        adj_mat_abs_fname = os.path.join(adj_mats_dir, adj_mat_fname)
        adj_mat = np.load(adj_mat_abs_fname)
        
        # The values of TE are underestimated, tho some of them could be negative.
        adj_mat[adj_mat < 0.] = 0.
        
        dfl = DirectedFlagComplex(adj_mat)
        dfl.set_path_graph_dim(n) 
        t_list, dim_HH_0_list, dim_HH_1_list, hh_list = dfl.comp_persistent_hh_char()
        dfl.save_homologies(
            os.path.join(res_dir, adj_mat_fname), 
            t_list, 
            dim_HH_0_list, 
            dim_HH_1_list
        )
            
            
if __name__ == '__main__':
    main()
