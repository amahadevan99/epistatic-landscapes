import numpy as np

# get index of node as integer, given format of tree (string)
def idx(x):
    return int(x.split('.')[0])
    
class Tree:
    def __init__(self):
        # indexes nodes by string of form index.epoch
        self.par_dict = {} # parent identity
        self.dist_dict = {} # distance to parent for newick (not the same as mutational distance)
        self.mut_dist_dict = {} # mutational distance to parent
        self.nodes = [] # does not include root
        self.lifetime_dict = {}
    
    def __len__(self):
        return len(self.nodes)
    
    # adds a leaf from the specified parent
    def add_node(self,child,parent):
        self.par_dict[child] = parent
        self.dist_dict[child] = 0
        self.nodes.append(child)
        if parent=='root' or idx(child)==idx(parent):
            self.mut_dist_dict[child] = 0
        else:
            self.mut_dist_dict[child] = 1
    
    # gets mutational distance between two lineages. Inputs are integers
    def get_mut_dist(self,n1,n2):
        par_dict = self.par_dict
        
        n2_par = {}
        x = '%s.%s'%(n2,n2)
        v = 0
        while True:
            n2_par[x] = v
            if idx(x) != idx(par_dict[x]): # checks if different lineage
                v += 1
            x = par_dict[x]
            if x=='root':break
                
        y = '%s.%s'%(n1,n1)
        u = 0
        while True:
            if y in n2_par:
                return u+n2_par[y]
            if idx(y) != idx(par_dict[y]):
                u += 1
            y = par_dict[y]
            
    # True if x is an ancestor of y. Inputs are integers
    def is_anc(self,x,y):
        u = '%s.%s'%(y,y)
        v = '%s.%s'%(x,x)
        while True:
            if u==v:
                return True
            if u=='root':
                return False
            u = self.par_dict[u]
            
    def get_child_dict(self,input_dict):
        # generate dictionary of list of children for every node
        inv_map = {}
        for k,v in input_dict.items():
            if v not in inv_map.keys():
                inv_map[v]=[k]
            else:
                inv_map[v].append(k)
        return inv_map
    
    def get_full_newick(self,par_dict=None,dist_dict=None,mutations=False):
        if par_dict is None:
            child_dict = self.get_child_dict(self.par_dict)
        else:
            child_dict = self.get_child_dict(par_dict)
        if dist_dict is None:
            if mutations:
                branch_lengths = self.mut_dist_dict
            else:
                branch_lengths = self.dist_dict
        else:
            branch_lengths = dist_dict

        def generate_newick(node):
            if node in child_dict:
                if node in branch_lengths:
                    dist = branch_lengths[node]
                else: # only for root
                    dist = 0
                return '('+(','.join([generate_newick(s) for s in child_dict[node]])
                           )+')'+str(node)+':%s'%dist
            else:
                return str(node)+':%s'%branch_lengths[node]
        return generate_newick('root')+';'

    def get_final_newick(self,node_list,mutations=False):
        init_node='root'
        def generate_newick(node):
            if node in final_child_dict:
                return '('+(','.join([generate_newick(s) for s in final_child_dict[node]])
                           )+')'+str(node)+':'+str(final_length_dict[node])
            else:
                return str(node)+':'+str(final_length_dict[node])

        final_child_dict,final_length_dict,_ = self.get_final_dicts(node_list,init_node=init_node,mutations=mutations)
        tree_string = generate_newick(init_node)+';'
        return tree_string


    # gets child dictionary and dictionary of branch lengths for final newick containing
    # nodes specified in node_list
    # only includes nodes which branch
    def get_final_dicts(self,node_list,init_node='root',mutations=False):
        par_dict = self.par_dict
        if mutations:
            branch_lengths = self.mut_dist_dict
        else:
            branch_lengths = self.dist_dict

        # sets entries of parent dictionary
        def set_parent_dict(node):
            if (node not in par_dict) or node in par_dict_intermediate:
                return
            par_dict_intermediate[node] = par_dict[node]
            set_parent_dict(par_dict[node])
            
        # gets length of branch and node name of closest upstream branch point (where final tree splits)
        def get_branch_length(node):
            if node == init_node:
                return (0,'scratch_node') # this node name doesn't matter and doesn't appear in final newick
            if par_dict_intermediate[node]==init_node or len(child_dict_intermediate[par_dict_intermediate[node]])>1:
                return (branch_lengths[node],par_dict_intermediate[node])
            xx = get_branch_length(par_dict_intermediate[node])
            return (branch_lengths[node]+xx[0],xx[1])

        par_dict_intermediate = {} # parent dictionary for final nodes and all their ancestors
        for child in node_list:
            set_parent_dict(child)
        child_dict_intermediate = self.get_child_dict(par_dict_intermediate)

        # the nodes that we want to keep are all those without exactly 1 child, plus the root node
        nodes_to_keep = [init_node,*[x for x in child_dict_intermediate if len(child_dict_intermediate[x])!=1],*node_list]
        
        final_parent_dict = {} # parents of final nodes
        final_length_dict = {} # branch lengths
        for nn in nodes_to_keep:
            final_length_dict[nn],final_parent_dict[nn] = get_branch_length(nn)
        
        final_child_dict = self.get_child_dict(final_parent_dict)
        return final_child_dict,final_length_dict,final_parent_dict


    # computes the number of generations for which each genotype was populated
    def compute_lifetime_dict(self):
        for n in self.nodes:
            dd = idx(n)
            if dd not in self.lifetime_dict:
                self.lifetime_dict[dd] = self.dist_dict[n]
            else:
                self.lifetime_dict[dd] += self.dist_dict[n]

    def get_reduced_newick(self,newick_reduction,mutations=False):
        if mutations:
            branch_lengths = self.mut_dist_dict
        else:
            branch_lengths = self.dist_dict

        if self.lifetime_dict=={}:
            self.compute_lifetime_dict()
        par_dict_reduced = {}
        dist_dict_reduced = {'0.0':0}

        # only includes genotypees which survived more than specified number of generations
        # if a node didn't live that long, reconnects back until parent lives long enough or root reached
        for s in self.par_dict:
            if self.lifetime_dict[idx(s)]<newick_reduction:
                continue
            par_dict_reduced[s] = self.par_dict[s]
            dist_dict_reduced[s] = branch_lengths[s]
            while not(par_dict_reduced[s]=='root') and self.lifetime_dict[idx(par_dict_reduced[s])]<newick_reduction:
                dist_dict_reduced[s] += branch_lengths[par_dict_reduced[s]]
                par_dict_reduced[s] = self.par_dict[par_dict_reduced[s]]

        # get rid of intermediate nodes which do not branch
        pointer_child_dict = self.get_child_dict(par_dict_reduced)
        pr_keys = list(par_dict_reduced.keys())
        for s in pr_keys:
            if s in pointer_child_dict and len(pointer_child_dict[s])==1 and not(s=='root'):
                dist_dict_reduced[pointer_child_dict[s][0]] += dist_dict_reduced[s]
                par_dict_reduced[pointer_child_dict[s][0]] = par_dict_reduced[s]
                par_dict_reduced.pop(s)
                dist_dict_reduced.pop(s)

        new = self.get_full_newick(par_dict=par_dict_reduced,dist_dict=dist_dict_reduced)
        return new

