import numpy as np
from scipy.stats import truncnorm


class Tree:
    def __init__(self):
        # indexes nodes by integers, but they can have labels stored in dictionary
        self.par_dict = {}
        self.nodes = [] # does not include root
    
    def __len__(self):
        return len(self.nodes)
    
    # adds a leaf from the specified parent
    def add_node(self,child,parent):
        self.par_dict[child] = parent
        self.nodes.append(child)
    
    # gets distance between two nodes
    def get_dist(self,n1,n2):
        par_dict = self.par_dict
        
        n2_par = {}
        x = n2
        v = 0
        while True:
            n2_par[x] = v
            x = par_dict[x]
            v +=1
            if x=='root':break
                
        y = n1
        u = 0
        while True:
            if y in n2_par:
                return u+n2_par[y]
            u += 1
            y = par_dict[y]
            
    # True if x is an ancestor of y
    def is_anc(self,x,y):
        u = y
        while True:
            if u==x:
                return True
            if u=='root':
                return False
            u = self.par_dict[u]
            
    def get_child_dict(self):
        # generate dictionary of list of children for every node
        inv_map = {}
        for k,v in self.par_dict.items():
            if v not in inv_map.keys():
                inv_map[v]=[k]
            else:
                inv_map[v].append(k)
        return inv_map
    
    def get_newick(self):
        child_dict = self.get_child_dict()
        def generate_newick(node):
            if node in child_dict:
                return '('+(','.join([generate_newick(s) for s in child_dict[node]])
                           )+')'+str(node)+':1'
            else:
                return str(node)+':1'
        return generate_newick('root')+';'






class RandomEvo:
    # stores the trajectory in a tree, where each node is associated with a mu and an s (in a tuple)
    def __init__(self,max_nodes,D_func,seed=None):
        if seed is None:
            self.seed = np.random.randint(2147483647)
        else:
            self.seed = seed
        np.random.seed(self.seed)
        self.tree = Tree() # each node of the tree has mu, and the s from its parent to it
        self.D_func = D_func # for correlations
        self.sig = np.sqrt(self.D_func(2)/2) # scale of s
        self.max_nodes = max_nodes # maximum number of nodes in tree
        self.cov_mat = np.zeros((2*max_nodes,2*max_nodes)) # covariance matrix of nodes that have been seen
        self.labels = {} # stores (mu,s) for each node, where s is from the parent
        self.active = {} # stores whether node is active
        self.fitness = {}
        
        self.cov_mat[0,0] = self.a_func(1)
        self.tree.add_node(0,'root')
        self.labels[0] = (0,0)
        self.active[0] = True
        self.fitness[0] = 0
    
    def a_func(self,n):
        return self.D_func(n) - 1/2*(self.D_func(n+1)+self.D_func(n-1))
    
    # mutates one active node
    def take_step(self):
        tree = self.tree
        maxn = self.max_nodes
        
        if len(tree)==maxn:
            print('max size exceeded')
            return
        
        sig = self.sig
        par = np.random.choice([n for n in tree.nodes if self.active[n]])
        
        mu_par = self.labels[par][0]
        
        # update s covariances for s to be added
        n = len(tree)
        self.cov_mat[n-1+maxn,:] = self.cov_mat[par,:]
        self.cov_mat[:,n-1+maxn] = self.cov_mat[:,par]
        self.cov_mat[n-1+maxn,n-1+maxn] = -self.a_func(0)
        
        s_child = mu_par + sig*truncnorm.rvs(-mu_par/sig,np.inf)
        past_values = [self.labels[x][0] for x in tree.nodes]
        past_values.extend([self.labels[x][1] for x in tree.nodes[1:]])
        past_values.append(s_child)
        
        # add node to tree (needed to update mu covariances)
        new_node = len(tree)
        tree.add_node(new_node,par)
        
        # update mu covariances
        self.cov_mat[n,n] = self.a_func(1)
        for i in range(n):
            self.cov_mat[n,i] = self.cov_mat[i,n] = self.get_mu_mu(n,i)
            self.cov_mat[n,i+maxn] = self.cov_mat[i+maxn,n] = self.get_mu_s(n,i)
        
        idxs = np.concatenate((np.arange(n),maxn+np.arange(n)))
        
        K = np.dot(self.cov_mat[n,idxs],np.linalg.inv(self.cov_mat[idxs,:][:,idxs]))
        cond_mean = np.dot(K,past_values)
        cond_cov = self.a_func(1) - np.dot(K,self.cov_mat[n,idxs].T)
        
        mu_child = cond_mean + np.sqrt(cond_cov)*np.random.randn()
        
        self.labels[new_node] = (mu_child,s_child)
        self.active[new_node] = True
        self.fitness[new_node] = self.fitness[par] + s_child
    
    def get_mean_fitness(self):
        return np.mean([self.fitness[x] for x in self.tree.nodes if self.active[x]])
    
    # deactivates nodes which are below mean fitness
    def cull_nodes(self):
        mean_fitness = self.get_mean_fitness()
        active_nodes = self.get_active_nodes()
        
        prob = np.array([np.exp((self.fitness[x]-mean_fitness)/100) for x in active_nodes])
        prob /= sum(prob)
        preserve = np.random.choice(active_nodes,size=5,p=prob)
        for x in active_nodes:
#             if self.fitness[x]<=.95*mean_fitness:
            if x not in preserve:
                self.active[x] = False
                
    def get_active_nodes(self):
        return [x for x in self.tree.nodes if self.active[x]]
        
    # covariance of mu at nodes x and y
    def get_mu_mu(self,x,y):
        tree = self.tree
        return self.a_func(tree.get_dist(x,y)+1)
    
    # covariance of mu at node x with s leading to node y+1
    def get_mu_s(self,x,y):
        tree = self.tree
        if not tree.is_anc(y+1,x):
            return self.a_func(tree.get_dist(x,y+1))
        else:
            return -self.a_func(tree.get_dist(x,y+1)+1)