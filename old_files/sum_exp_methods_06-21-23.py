import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm
from tree_methods import Tree


# methods for sampling multivariate gaussian with weighted sum constraint (Vrins 2018)
# samples Z vector of independent gaussians conditions on dot(Z,nu)>ups
def sample_truncated(nu,ups):
    n = len(nu)
    norm_nu = np.sqrt(np.sum(nu**2))

    if ups/norm_nu<3:
        u = np.random.random()
        cbar = norm_nu*norm.ppf((1-u)+u*norm.cdf(ups/norm_nu))
    else:
        cbar = ups + norm_nu**2/ups*np.random.exponential() # approximation for gaussian
    return sample_c(nu,cbar,norm_nu=norm_nu)

def sample_c(nu,c,norm_nu=None):
    n = len(nu)
    if norm_nu is None:
        norm_nu = np.sqrt(np.sum(nu**2))

    if n==1:
        return np.array([c/norm_nu])

    mu = c*nu**2/norm_nu**2
    cov_mat = -np.outer(nu**2,nu**2)
    for i in range(n):
        cov_mat[i,i] = nu[i]**2*(norm_nu**2-nu[i]**2)
    cov_mat = cov_mat/norm_nu**2

    x_vec = np.random.multivariate_normal(mu[:-1],cov_mat[:-1,:-1])
    x_vec = np.append(x_vec,c-np.sum(x_vec))
    return x_vec/nu

# get index of node as integer, given format of tree (string)
def idx(x):
    return int(x.split('.')[0])

####################################
####################################

# runs evolution where D(l) is a sum of exponentials: therefore does not store entire covariance matrix
# still assumes treelike limit
class RandomEvo_Exp:
    # stores the trajectory in a tree, where each node is associated with a mu and an s (in a tuple)
    def __init__(self,coeff_list,corr_len_list,seed=None):
        if seed is None:
            self.seed = np.random.randint(2147483647)
        else:
            self.seed = seed
        np.random.seed(self.seed)
        self.tree = Tree() # each node of the tree has mu, and the s from its parent to it
        self.coeff_list = coeff_list # list of coefficients for each exponential
        self.corr_len_list = corr_len_list # correlation lengths for each exponential
        assert len(coeff_list)==len(corr_len_list)

        self.rho_list = np.exp(-1/corr_len_list)
        self.rho_func = np.sqrt((1-self.rho_list**2)/2)
        self.K = len(coeff_list) # number of exponentials to track

        self.labels = {} # stores ([mu_1,mu_2,..],[s_1,s_2,..]) for each node
        self.active = {} # stores whether node is active
        self.fitness = {}
        
        self.tree.add_node('0.0','root') # parents are denoted by index.epoch
        self.labels['0.0'] = (np.zeros(self.K),np.zeros(self.K))
        self.active['0.0'] = True
        self.fitness['0.0'] = 0

        self.epoch = 0 # number of evolutionary steps taken
    
    # mutates one active node
    def take_step(self):
        tree = self.tree
        
        par = np.random.choice([n for n in tree.nodes if self.active[n]])
        mu_par = self.labels[par][0]

        # should the conditioning be that new fitness is greater than mean fitness?
        random_part = sample_truncated(self.rho_func*self.coeff_list,-np.dot(mu_par,self.coeff_list))
        s_child = mu_par + self.rho_func*random_part
        assert np.dot(self.coeff_list,s_child)>=0
        mu_child = mu_par - (1-self.rho_list)*s_child

        # grow nodes in tree
        for x in self.tree.nodes:
            if self.active[x]:
                self.tree.dist_dict[x]+=1

        # add node to tree
        self.epoch += 1
        new_node = '%s.%s'%(self.epoch,self.epoch)
        new_par = '%s.%s'%(idx(par),self.epoch)
        tree.add_node(new_node,par)
        tree.add_node(new_par,par)
        
        self.labels[new_node] = (mu_child,s_child)
        self.active[new_node] = True
        self.fitness[new_node] = self.fitness[par] + np.dot(self.coeff_list,s_child)#np.sum(s_child)

        self.labels[new_par] = self.labels[par]
        self.active[new_par] = True
        self.active[par] = False
        self.fitness[new_par] = self.fitness[par]
    
    def get_mean_fitness(self):
        return np.mean([self.fitness[x] for x in self.tree.nodes if self.active[x]])
    
    # deactivates nodes according to protocol of choice
    def cull_nodes(self):
        mean_fitness = self.get_mean_fitness()
        active_nodes = self.get_active_nodes()
        
        prob = np.array([np.exp(.1*(self.fitness[x]-mean_fitness)) for x in active_nodes])
        # sorted_nodes = [active_nodes[x] for x in np.argsort(prob)]
        # prob /= sum(prob)
        for kk,x in enumerate(active_nodes):
            # if self.fitness[x]<=.9*mean_fitness:
            # if x not in preserve:
            # if x not in sorted_nodes[-10:]:
            if np.random.random()>prob[kk]:
                self.active[x] = False
                
    def get_active_nodes(self):
        return [x for x in self.tree.nodes if self.active[x]]
        
    # runs evolution until max number of nodes is saturated
    def run_evo(self,num_steps):
        self.s_dist_traj = [] # mean fitness of population over evolution
        self.num_active_traj = [] # number of active walkers over evolution
        self.ds_traj = [] # fitness gains
        self.mu_traj = []
        for i in range(num_steps):
            self.take_step()
            self.cull_nodes()
            self.s_dist_traj.append([self.fitness[x] for x in self.get_active_nodes()])
            self.num_active_traj.append(np.sum(list(self.active.values())))
            self.ds_traj.append(np.dot(self.coeff_list,self.labels['%s.%s'%(i,i)][1]))
            self.mu_traj.append(np.dot(self.coeff_list,self.labels['%s.%s'%(i,i)][0]))

    def get_full_newick(self,mutations=False):
        return self.tree.get_full_newick(mutations=mutations)

    # get newick where leaves are active nodes
    def get_final_newick(self,mutations=False):
        return self.tree.get_final_newick(self.get_active_nodes(),mutations=mutations)

