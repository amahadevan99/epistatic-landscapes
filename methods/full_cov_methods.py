import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm
from tree_methods import Tree

# get index of node as integer, given format of tree (string)
def idx(x):
    return int(x.split('.')[0])

# stores entire covariance matrix; assumes treelike limit
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
        self.tree.add_node('0.0','root')
        self.labels['0.0'] = (0,0)
        self.active['0.0'] = True
        self.fitness['0.0'] = 0
        self.epoch = 0 # number of evolution steps taken
    
    def a_func(self,n):
        return self.D_func(n) - 1/2*(self.D_func(n+1)+self.D_func(n-1))
    
    # mutates one active node
    def take_step(self):
        tree = self.tree
        maxn = self.max_nodes
        
        sig = self.sig

        # sorting the parent choices gives agreement for the same random seed.
        # otherwise small differences can cause a difference in the parent which causes big differences later on
        par_choices = [n for n in tree.nodes if self.active[n]]
        par_choice_idxs = [idx(n) for n in tree.nodes if self.active[n]]
        par = np.random.choice(np.array(par_choices)[np.argsort(par_choice_idxs)])
        par_idx = idx(par)
        
        mu_par = self.labels[par][0]
        
        # update s covariances for s to be added
        self.epoch += 1
        n = self.epoch
        self.cov_mat[n-1+maxn,:] = self.cov_mat[par_idx,:]
        self.cov_mat[:,n-1+maxn] = self.cov_mat[:,par_idx]
        self.cov_mat[n-1+maxn,n-1+maxn] = -self.a_func(0)
        
        s_child = mu_par + sig*truncnorm.rvs(-mu_par/sig,np.inf)
        past_values = [self.labels['%s.%s'%(x,x)][0] for x in range(self.epoch)]
        past_values.extend([self.labels['%s.%s'%(x,x)][1] for x in range(1,self.epoch)])
        past_values.append(s_child)

        # grow nodes in tree
        for x in self.tree.nodes:
            if self.active[x]:
                self.tree.dist_dict[x]+=1

        # add node to tree
        new_node = '%s.%s'%(self.epoch,self.epoch)
        new_par = '%s.%s'%(idx(par),self.epoch)
        tree.add_node(new_node,par)
        tree.add_node(new_par,par)

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

        self.labels[new_par] = self.labels[par]
        self.active[new_par] = True
        self.active[par] = False
        self.fitness[new_par] = self.fitness[par]
    
    def get_mean_fitness(self):
        return np.mean([self.fitness[x] for x in self.tree.nodes if self.active[x]])
    
    # deactivates nodes which are below mean fitness
    def cull_nodes(self):
        mean_fitness = self.get_mean_fitness()
        active_nodes = self.get_active_nodes()
        
        prob = np.array([np.exp(self.fitness[x]) for x in active_nodes])
        sorted_nodes = [active_nodes[x] for x in np.argsort(prob)]
        prob /= sum(prob)
        for x in active_nodes:
            if self.fitness[x]<=.9*mean_fitness:
            # if x not in preserve:
            # if x not in sorted_nodes[-4:]:
                self.active[x] = False
                
    def get_active_nodes(self):
        return [x for x in self.tree.nodes if self.active[x]]
        
    # covariance of mu at nodes x and y
    def get_mu_mu(self,x,y):
        tree = self.tree
        return self.a_func(tree.get_mut_dist(x,y)+1)
    
    # covariance of mu at node x with s leading to node y+1
    def get_mu_s(self,x,y):
        tree = self.tree
        if not tree.is_anc(y+1,x):
            return self.a_func(tree.get_mut_dist(x,y+1))
        else:
            return -self.a_func(tree.get_mut_dist(x,y+1)+1)

    # runs evolution until max number of nodes is saturated
    def run_evo(self):
        self.mean_s_traj = [] # mean fitness of population over evolution
        self.num_active_traj = [] # number of active walkers over evolution
        while self.epoch<self.max_nodes-1:
            self.take_step()
            self.cull_nodes()
            self.mean_s_traj.append(self.get_mean_fitness())
            self.num_active_traj.append(np.sum(list(self.active.values())))
