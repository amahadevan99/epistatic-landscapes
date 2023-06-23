import numpy as np
from scipy.stats import norm
# from scipy.stats import truncnorm
import tree_methods as tm

# get index of node as integer, given format of tree (string)
def idx(x):
    return int(x.split('.')[0])

####################################
####################################

# runs evolution where D(l) is a sum of exponentials: therefore does not store entire covariance matrix
# still assumes treelike limit
# resolves the population at the level of fitness classes
class RandomEvo_Exp:
    # stores the trajectory in a tree, where each node is associated with a mu and an s (in a tuple)
    def __init__(self,N,U,coeff_list,corr_len_list,additive=1,seed=None):
        if seed is None:
            self.seed = np.random.randint(2147483647)
        else:
            self.seed = seed
        np.random.seed(self.seed)

        self.N = N # population size
        self.U = U # overall mutation rate (per generation)
        self.additive = additive # coefficient of additive piece
        self.tree = tm.Tree() # each node of the tree is a fitness class, with a mu, and the s from its parent to it
        self.coeff_list = coeff_list # list of coefficients for each exponential
        self.corr_len_list = corr_len_list # list of correlation lengths for each exponential
        assert len(coeff_list)==len(corr_len_list)

        self.rho_list = np.exp(-1/corr_len_list)
        self.rho_func = np.sqrt((1-self.rho_list**2)/2)
        self.K = len(coeff_list) # number of exponentials to track

        self.labels = {} # stores ([mu_1,mu_2,..],[s_1,s_2,..]) for each node
        self.subpop = {} # stores number of individuals in each fitness class
        self.fitness = {}
        
        self.tree.add_node('0.0','root') # parents are denoted by index.epoch
        self.labels['0.0'] = (np.zeros(self.K),np.zeros(self.K))
        self.subpop['0.0'] = N
        self.fitness['0.0'] = 0
        self.active_nodes = ['0.0'] # list of fitness classes which are populated

        self.count = 0 # number of fitness classes instantiated
        self.gen = 0 # generation number of evolution
    
    # grows all nodes in the tree corresponding to populated fitness classes
    def grow_tree(self):
        for x in self.active_nodes:
            self.tree.dist_dict[x]+=1

    # makes k new fitness classes, descended from a specified parent fitness class
    def new_fitness_classes(self,par,k):
        if k==0:return
        tree = self.tree
        mu_par = self.labels[par][0]

        for child_idx in range(k):
            s_child = mu_par + self.rho_func*np.random.randn(self.K)
            mu_child = mu_par - (1-self.rho_list)*s_child

            # add node to tree
            self.count += 1
            new_node = '%s.%s'%(self.count,self.gen)
            tree.add_node(new_node,par)
            
            self.labels[new_node] = (mu_child,s_child)
            self.subpop[new_node] = 1
            self.active_nodes.append(new_node)
            self.fitness[new_node] = self.fitness[par]+np.dot(self.coeff_list,s_child)+np.sqrt(self.additive)*np.random.randn()
        
        new_par = '%s.%s'%(idx(par),self.gen)
        tree.add_node(new_par,par)
        self.labels[new_par] = self.labels[par]
        self.subpop[new_par] = self.subpop[par]
        self.subpop[par] = 0
        self.fitness[new_par] = self.fitness[par]
        self.active_nodes.remove(par)
        self.active_nodes.append(new_par)

    def get_mean_fitness(self):
        return np.mean([self.fitness[x] for x in self.active_nodes])
    
    def step_generation(self):
        self.gen += 1
        self.grow_tree()

        mean_fitness = self.get_mean_fitness()
        active_nodes = self.active_nodes

        mean_offspring = np.array([self.subpop[x]*np.exp(self.fitness[x]-mean_fitness) for x in active_nodes])
        actual_offspring = np.random.multinomial(self.N,mean_offspring/np.sum(mean_offspring))
        num_muts = np.random.binomial(actual_offspring,self.U)

        for kk,an in enumerate(active_nodes.copy()): # since active_nodes changes during iteration
            self.subpop[an] = actual_offspring[kk] - num_muts[kk]
            self.new_fitness_classes(an,num_muts[kk])

        for x in active_nodes:
            if self.subpop[x]==0:
                active_nodes.remove(x)
   
    # runs evolution for desired number of generations
    def run_evo(self,num_steps):
        self.s_dist_traj = [] # mean fitness of population over evolution
        self.num_active_classes = [] # number of populated fitness classes over evolution
        self.mean_fitness = []
        for i in range(num_steps):
            self.step_generation()
            an = self.active_nodes
            self.s_dist_traj.append({self.fitness[x]:self.subpop[x] for x in an})
            self.num_active_classes.append(len(an))
            self.mean_fitness.append(self.get_mean_fitness())

    def get_full_newick(self,mutations=False):
        return self.tree.get_full_newick(mutations=mutations)

    # get newick where leaves are active nodes
    def get_final_newick(self,mutations=False):
        return self.tree.get_final_newick(self.active_nodes,mutations=mutations)

    def get_reduced_newick(self,newick_reduction,mutations=False):
        return self.tree.get_reduced_newick(newick_reduction,mutations=mutations)


