import numpy as np
import scipy.sparse as sp
import itertools
import functools
import networkx as nx
from typing import Tuple, List, Dict
from tqdm import tqdm
import os


class DirectedFlagComplex:
    def __init__(
        self, 
        w_adj_m: np.ndarray
    ) -> None:
        """
        Args:
            w_adj_m: the weighted adjacency matrix of the original graph.
        """
        self.w_adj_m = w_adj_m
        self.vertices = list(range(w_adj_m.shape[0]))
        self.n_simplices = None # list of n-simplices for the current step of the filtration
        self.simplex_vertice_mat = None # NumPy
        
        self.conn_g_adj_m_prev = None
        self.conn_g_adj_m_cur = None
        
        self.conn_g_prev = None
        self.conn_g_cur = None
        
        self.condensation_prev = None
        self.condensation_cur = None
        
        self.condensation_paths = {}
        
        self.new_n_vertices = None
        self.new_n_edges = None
        self.dim_HH_0_cur = 0
        self.dim_HH_1_cur = 0
        
        
    def set_path_graph_dim(
        self, 
        dim: int
    ) -> None:
        """
        Sets the dimensionality of n-path digraph that will be constructed.
        
        Args:
            dim: the dimensionality of n-path digraph.
        """
        self.n = dim
        
        
    def get_dfl(
        self, 
        adj_mat: sp.dok_matrix, 
        max_dim: int
    ) -> List[Tuple[int]]:
        """
        Computes the directed flag complex for the given graph 
        up to the given dimension.
        
        Args:
            adj_mat: binary adjacency matrix of the graph.
            max_dim: the maximum dimension of simplices for the complex.
            
        Returns:
            The constructed directed flag complex.
        """
        vertices_num = adj_mat.shape[0]
        dfl = [[] for i in range(max_dim + 1)]
        dfl[0] = [(v,) for v in range(vertices_num)]
        for d in range(1, max_dim + 1):
            # takes (d-1)-simplex (v_1, ..., v_(d - 1)) and 
            # constructs d-simplices of the form (v_1, ..., v_(d - 1), v)
            # for all v such that exists an edge (v_i, v), i=1, ..., (d - 1)
            dfl[d] = [
                simplex + (v,) 
                for simplex in dfl[d - 1] 
                for v in np.where(np.bitwise_and.reduce(adj_mat[list(simplex)], axis=0))[0]
            ]
        return dfl
        
    
    def get_simplex_vertice_mat(
        self, 
        vertices: List[int], 
        simplices: List[Tuple[int]]
    ) -> sp.dok_matrix:
        """
        Computes the adjacency matrix for the given lists of simplices
        and vertices.
        
        Args:
            vertices: list of vertices.
            simplices: list of simplices.
            
        Returns: adjacency matrix A such that A[i, j] = I(vertex_j \in simplex_i). 
        """
        vertices_num = len(vertices)
        simplices_num = len(simplices)
        
        simplex_vertice_mat = sp.dok_matrix((simplices_num, vertices_num), dtype=int)
        for simplex_idx, simplex in enumerate(simplices):
            for v in simplex:
                simplex_vertice_mat[simplex_idx, v] = 1
                
        return simplex_vertice_mat.copy()
                
    
    def get_n_path_g(
        self, 
        vertices: List[int], 
        n_simplices: List[Tuple[int]]
    ) -> nx.DiGraph:
        """
        Computes the n-path digraph.
        
        Args:
            vertices: list of vertices.
            n_simplices: list of n-simplices.
        
        Returns:
            The constructed n-path digraph.
        """
        n = self.n
        
        simplex_vertice_mat = self.get_simplex_vertice_mat(vertices, n_simplices).tocsr()
        self.simplex_vertice_mat = simplex_vertice_mat
        
        n_path_adj_mat = (simplex_vertice_mat @ simplex_vertice_mat.T == n)
        simplex_1_idxs, simplex_2_idxs = n_path_adj_mat.nonzero()
        
        # Each simplex (v_0, ..., v_n) is encoded 
        # with bit array 2 ** v_0 + ... 2 ** v_n.
        simplex_1_bitarrs = np.array(
            [sum(map(lambda v: 1 << v, n_simplices[idx])) \
            for idx in simplex_1_idxs], dtype=np.int64
        )
        simplex_2_bitarrs = np.array(
            [sum(map(lambda v: 1 << v, n_simplices[idx])) \
            for idx in simplex_2_idxs], dtype=np.int64
        )
        
        # i > j for d_i(sigma), d_j(tau) <=> 
        # <=> sigma xor (sigma and tau) > tau xor (sigma and tau) <=> 
        # <=> sigma & not tau > not sigma & tau  
        adj_simplices = np.where((simplex_1_bitarrs[..., None] & ~simplex_2_bitarrs) \
                            > (~simplex_1_bitarrs[..., None] & simplex_2_bitarrs))
        simplex_1_idxs = simplex_1_idxs[adj_simplices[0]]
        simplex_2_idxs = simplex_2_idxs[adj_simplices[1]]
        n_path_adj_mat = sp.csr_matrix(
            ([1] * adj_simplices[0].shape[0], (simplex_1_idxs, simplex_2_idxs)), 
            shape=n_path_adj_mat.shape
        )
        
        return nx.DiGraph(n_path_adj_mat), n_path_adj_mat
        
    
    def update_n_path_g(
        self
    ) -> None:
        """
        Updates the n-path digraph and its adjacency matrix.
        """
        self.conn_g_adj_m_prev = self.conn_g_adj_m_cur
        self.conn_g_prev = self.conn_g_cur
         
        n = self.n
        n_simplices = self.n_simplices.copy()
        old_simplices_num = len(n_simplices)
        n_simplices += self.new_n_vertices
        
        new_simplex_vertice_mat = self.get_simplex_vertice_mat(
            self.vertices, 
            self.new_n_vertices
        ).tocsr()
        simplex_vertice_mat = sp.vstack((self.simplex_vertice_mat, new_simplex_vertice_mat))
        self.simplex_vertice_mat = simplex_vertice_mat
        
        n_path_adj_mat = (simplex_vertice_mat @ simplex_vertice_mat.T == n)
        simplex_1_idxs, simplex_2_idxs = n_path_adj_mat.nonzero()
        
        # Each simplex (v_0, ..., v_n) is encoded 
        # with bit array 2 ** v_0 + ... 2 ** v_n.
        simplex_1_bitarrs = np.array(
            [sum(map(lambda v: 1 << v, n_simplices[idx])) \
            for idx in simplex_1_idxs], dtype=np.int64
        )
        simplex_2_bitarrs = np.array(
            [sum(map(lambda v: 1 << v, n_simplices[idx])) \
            for idx in simplex_2_idxs], dtype=np.int64
        )
        
        # i > j for d_i(sigma), d_j(tau) <=> 
        # <=> sigma xor (sigma and tau) > tau xor (sigma and tau) <=> 
        # <=> sigma & not tau > not sigma & tau  
        adj_simplices = np.where(
            (simplex_1_bitarrs[..., None] & ~simplex_2_bitarrs) \
            > (~simplex_1_bitarrs[..., None] & simplex_2_bitarrs)
        )
        simplex_1_idxs = simplex_1_idxs[adj_simplices[0]]
        simplex_2_idxs = simplex_2_idxs[adj_simplices[1]]
        n_path_adj_mat = sp.csr_matrix(
            (
                [1] * adj_simplices[0].shape[0], 
                (simplex_1_idxs, simplex_2_idxs)
            ), 
            shape=n_path_adj_mat.shape
        )
        
        new_edges = [
            (simplex_1, simplex_2) 
            for simplex_1, simplex_2 
            in zip(simplex_1_idxs, simplex_2_idxs) 
            if simplex_1 >= old_simplices_num or simplex_2 >= old_simplices_num
        ]
        
        self.new_n_edges = new_edges
        
        # Updating the current connectivity digraph and its adjacency matrix.
        self.conn_g_adj_m_cur = n_path_adj_mat
        self.conn_g_cur = nx.DiGraph(n_path_adj_mat)
        
        
    def update_new_n_simplices(
        self, 
        new_edge: Tuple[int]
    ) -> None:
        """
        Updates the list of directed n-simplices 
        after adding the new edge to the original digraph.
        
        Args:
            new_edge: the edge added to the digraph.
        """
        new_v_1, new_v_2 = new_edge
        
        # An almost n-simplex is a pair ({s, s'}, new_edge),
        # where s and s' - (n-1)-simplices,
        #       new_edge = (v_i, v'_i'),
        #       d_i(s) = d_i'(s')
        #
        # possible_subsimplices_1 := {(s, i) | s \in dfl[n - 1], new_edge[0] = s_i}
        # possible_subsimplices_2 := {(s', i') | s' \in dfl[n - 1], new_edge[1] = s'_i'}
        #
        # v'_i' was on the pos_2-th position in s' =>
        # => it will be on the (pos_2 + 1)_th position in the new simplex.
        possible_subsimplices_1 = [
            (simplex_1, simplex_1.index(new_v_1)) 
            for simplex_1 in self.dfl_t[self.n - 1] 
            if new_v_1 in simplex_1
        ]
        possible_subsimplices_2 = [
            (simplex_2, simplex_2.index(new_v_2)) 
            for simplex_2 in self.dfl_t[self.n - 1] 
            if new_v_2 in simplex_2
        ]
        
        new_n_simplices = []
        for (subsimplex_1, pos_1), (subsimplex_2, pos_2) in itertools.product(possible_subsimplices_1, possible_subsimplices_2):
            if set(subsimplex_1) - {subsimplex_1[pos_1]} == set(subsimplex_2) - {subsimplex_2[pos_2]}:
                new_n_simplex = subsimplex_1[: pos_2 + 1] + (new_v_2, ) + subsimplex_1[pos_2 + 1 :]
                new_n_simplices.append(new_n_simplex)
        
        self.new_n_vertices = new_n_simplices
        
        
    def update_condensation(
        self
    ) -> None:
        """
        Updates the condensation of the n-path digraph.
        """
        # ASSUME n >= 2!!!
        
        # The n-path graph is non-empty for the first time 
        if self.condensation_cur is None and len(self.new_n_vertices):
            self.condensation_cur = nx.condensation(self.conn_g_cur)
            self.condensation_paths = {
                e: list(nx.all_simple_paths(self.condensation_cur, e[0], e[1]))
                for e in self.condensation_cur.edges()
            }
            return
            
        # The condensation was computed at the previous step.
        # Need only to recompute it.
        self.condensation_prev = self.condensation_cur
        
        # IF n >= 2:
        #   1) !new_n_vertices & !new_n_edges => not to change
        #   2) !new_n_vertices & new_n_edges => impossible
        #   3) new_n_vertices & !new_n_edges => dim_HH_0 += len(new_n_vertices)
        #   4) new_n_vertices & new_n_edges => full recomputation
        if len(self.new_n_vertices):
            # Adding the new vertices
            self.conn_g_cur.add_nodes_from(
                range(
                    self.conn_g_cur.number_of_nodes(),
                    self.conn_g_cur.number_of_nodes() + len(self.new_n_vertices)
                )
            )
            
            new_n_vertices_idxs = range(
                self.condensation_cur.number_of_nodes(), 
                self.condensation_cur.number_of_nodes() + len(self.new_n_vertices)
            )
            new_condensation_cur_vertices_idxs = range(
                self.condensation_cur.number_of_nodes(), 
                self.condensation_cur.number_of_nodes() + len(self.new_n_vertices)
            )
            self.condensation_cur.add_nodes_from(new_n_vertices_idxs)
            self.condensation_cur.graph['mapping'].update(
                {
                    v_idx: c_idx 
                    for v_idx, c_idx in 
                    zip(new_n_vertices_idxs, new_condensation_cur_vertices_idxs)
                }
            )
            for v_idx, c_idx in zip(new_n_vertices_idxs, new_condensation_cur_vertices_idxs):
                self.condensation_cur.nodes()[c_idx].update({'members': {v_idx}})
            
            # Adding the new edges
            if len(self.new_n_edges):
                weak_comps = list(nx.weakly_connected_components(self.conn_g_cur))
                
                conn_g_adj_m_before_upd = self.conn_g_adj_m_cur
                conn_g_adj_m_before_upd[: -len(self.new_n_vertices)] = 0
                conn_g_adj_m_before_upd[:, : -len(self.new_n_vertices)] = 0
                for e_new in self.new_n_edges:
                    scc_map_cur = self.condensation_cur.graph['mapping']
                    
                    v1, v2 = e_new
                    weak_comp_idx_1 = [
                        i for i, weak_comp in enumerate(weak_comps) 
                        if v1 in weak_comp
                    ][0]
                    weak_comp_idx_2 = [
                        i for i, weak_comp in enumerate(weak_comps) 
                        if v2 in weak_comp
                    ][0]
                    
                    if weak_comp_idx_1 != weak_comp_idx_2:
                        # v1 and v2 are stored in vertices of condensation
                        # that are not connected with an edge.
                        #
                        # The only change to condensation is the edge adding.
                        #
                        # Only 1 path must be added: (v1, v2)
                        
                        new_condensation_edge = (scc_map_cur[v1], scc_map_cur[v2])
                        self.condensation_cur.add_edge(*new_condensation_edge)
                        weak_comps.append(weak_comps[weak_comp_idx_1] | weak_comps[weak_comp_idx_2])
                        weak_comps.pop(weak_comp_idx_1)
                        weak_comps.pop(
                            weak_comp_idx_2 
                            if weak_comp_idx_2 < weak_comp_idx_1 
                            else weak_comp_idx_2 - 1
                        )

                        self.condensation_paths.update(
                            {new_condensation_edge: [new_condensation_edge]}
                        )
                        
                    elif scc_map_cur[v1] != scc_map_cur[v2]:
                        # v1 and v2 are stored in vertices of condensation
                        # that are connected with an edge.
                        # This edge corresponds to one or more one-directed edges
                        # in the n-path graph.
                        #
                        # Need to change condensation only if the direction of (v1, v2)
                        # is opposite to ones for these edges.
                        #
                        # 1) not v1->v2 and not v2->v1 => add_edge
                        # 2) not v1->v2 and     v2->v1 => compute cycles
                        # 3)     v1->v2 and not v2->v1 => no changes
                        # 4)     v1->v2 and     v2->v1 => impossible
                        # It is comfortably to handle at first 2) and then 1).
                        
                        if nx.has_path(self.condensation_cur, scc_map_cur[v2], scc_map_cur[v1]):
                            
                            new_condensation_edge = (scc_map_cur[v1], scc_map_cur[v2])
                            self.condensation_cur.add_edge(*new_condensation_edge)
                            
                            new_wcc = weak_comps[weak_comp_idx_1] | weak_comps[weak_comp_idx_2]
                            appeared_cycles = nx.simple_cycles(self.condensation_cur.subgraph(new_wcc))
                            new_scc = set(functools.reduce(lambda x, y: x + y, appeared_cycles))
                            
                            new_scc_idx = self.condensation_cur.number_of_nodes()
                            
                            # Updating the condensation edges
                            edges_from_new_scc = [
                                (new_scc_idx, v2) 
                                for (v1, v2) in self.condensation_cur 
                                if v1 in new_scc and v2 not in new_scc
                            ]
                            edges_to_new_scc = [
                                (v1, new_scc_idx) 
                                for (v1, v2) in self.condensation_cur 
                                if v1 not in new_scc and v2 in new_scc
                            ]
                            self.condensation_cur.add_edges_from(edges_from_new_scc)
                            self.condensation_cur.add_edges_from(edges_to_new_scc)
                            
                            # Updating the condensation nodes
                            self.condensation_cur.remove_nodes_from(new_scc)
                            self.condensation_cur.add_node(new_scc_idx)
                            
                            # Updating the condensation mapping
                            self.condensation_cur.graph['mapping'].update(
                                {v: new_scc_idx for v in new_scc}
                            )
                            
                            ## Updating the condensation scc members - IS NOT USED!
                            #self.condensation_cur.nodes[i]
                            
                            # Updating the condensation paths
                            self.condensation_paths.update(
                                {
                                    e: nx.all_simple_paths(self.condensation_cur, e[0], e[1])
                                    for path in self.condensation_paths[e]
                                    for v1, v2 in zip(path[:-1], path[1:])
                                    if v1 in new_scc and v2 in new_scc
                                }
                            )
                        
                        elif not nx.has_path(self.condensation_cur, scc_map_cur[v1], scc_map_cur[v2]):
                            new_condensation_edge = (scc_map_cur[v1], scc_map_cur[v2])
                            self.condensation_cur.add_edge(*new_condensation_edge)
                            self.condensation_paths.update(
                                {new_condensation_edge: [new_condensation_edge]}
                            )
                                               
                    conn_g_adj_m_before_upd[v1, v2] = 1
    
    
    
    def update_hochschild_homologies(
        self
    ) -> None:
        """
        Updates the dimensionality of the 0-th and 1-st Hochschild homology groups.
        """
        if self.condensation_cur is not None:
            self.dim_HH_0_cur = nx.number_weakly_connected_components(
                self.condensation_cur
            )
            self.dim_HH_1_cur = self.dim_HH_0_cur \
                                - self.condensation_cur.number_of_nodes() \
                                + sum(map(len, self.condensation_paths.values()))
    
    
    def comp_persistent_hh_char(
        self
    ) -> Tuple[List]:
        """
        Computes the dimensionalities of the 0-th and 1-st Hochschild homology groups
        and Hochschild characteristics in the persistence pipeline.
        
        Returns:
            t_list: the list of filtration levels.
            dim_HH_0_list: the list of dimensionalities of the HH_0
            dim_HH_1_list: the list of dimensionalities of the HH_1
            hh_char_list: the list of Hochschild characteristics.
        """
        w_adj_m = self.w_adj_m
        
        hh_char_list = []
        dim_HH_0_list = []
        dim_HH_1_list = []
        t_list = np.sort(np.unique(w_adj_m[w_adj_m != 0.]))[::-1] 
        
        t_pos_dict = {t: np.where(w_adj_m >= t) for t in t_list}
        t_edge_dict = {t: np.where(w_adj_m == t) for t in t_list}
        for t in t_edge_dict:
            v0, v1 = t_edge_dict[t]
            t_edge_dict[t] = (v0[0], v1[0])
            
        self.adj_m_t = np.full(w_adj_m.shape, False)                                                  
        
        self.adj_m_t[t_pos_dict[t_list[0]]] = True
        self.dfl_t = self.get_dfl(self.adj_m_t, self.n)
        self.n_simplices = self.dfl_t[self.n]
        self.new_n_vertices = self.n_simplices
        self.new_n_edges = []
        self.conn_g_cur, self.conn_g_adj_m_cur = self.get_n_path_g(
            self.dfl_t[0], 
            self.dfl_t[self.n]
        )
        self.update_condensation()
        self.update_hochschild_homologies()
        
        dim_HH_0_list.append(self.dim_HH_0_cur)
        dim_HH_1_list.append(self.dim_HH_1_cur)
        hh_char_list.append(self.dim_HH_0_cur - self.dim_HH_1_cur)

        for t in tqdm(t_list[1:]):
            self.adj_m_t[t_pos_dict[t]] = True
            self.dfl_t = self.get_dfl(self.adj_m_t, self.n)
            new_edge = t_edge_dict[t] 
            self.update_new_n_simplices(new_edge) 
            if self.new_n_vertices:
                self.update_n_path_g()
                self.n_simplices += self.new_n_vertices
                self.update_condensation()
                self.update_hochschild_homologies()
            
                dim_HH_0_list.append(self.dim_HH_0_cur)
                dim_HH_1_list.append(self.dim_HH_1_cur)
                hh_char_list.append(self.dim_HH_0_cur - self.dim_HH_1_cur)
            
        return t_list, dim_HH_0_list, dim_HH_1_list, hh_char_list
        
    
    def save_homologies(
        self, 
        result_dir: str, 
        t_list: List[float], 
        dim_HH_0_list: List[int], 
        dim_HH_1_list: List[int]
    ) -> None:
        """
        Saves the lists of filtration levels, dim(HH_0), dim(HH_1) in .npy format.
        
        Args:
            result_dir: the directory to save the results to.
            t_list: the list of filtration levels.
            dim_HH_0_list: the list of dimensionalities of the HH_0
            dim_HH_1_list: the list of dimensionalities of the HH_1
        """
        if os.path.exists(result_dir):
            raise f"The directory exists: {result_dir}"
        
        os.mkdir(result_dir)    
        np.save(os.path.join(result_dir, 't_list'), np.array(t_list))
        np.save(os.path.join(result_dir, 'dim_HH_0_list'), np.array(dim_HH_0_list))
        np.save(os.path.join(result_dir, 'dim_HH_1_list'), np.array(dim_HH_1_list))
