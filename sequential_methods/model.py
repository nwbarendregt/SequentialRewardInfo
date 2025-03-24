"""
model.py
Contains class objects for different decision models (rewardmax and infomax).
Specific models are defined as subclasses of the Model superclass.
"""
import numpy as np
from abc import ABC, abstractmethod # Used for creating Model superclass
"""
Model():
Superclass that dictates the structural requirements of each model subclass.
"""
class Model(ABC):
    def __init__(self):
        pass
    """
    define_objective():
    Abstract method that defines the objective function each model seeks to optimize.
    Specific objectives are defined in a specific model's subclass.
    """
    @abstractmethod
    def define_objective(self, task_design):
        raise NotImplementedError()
"""
RewardMax():
Model subclass that defines the reward-maximizing strategy.
"""        
class RewardMax(Model):
    def __init__(self, discounting):
        super().__init__()
        self.gamma = discounting # Parameter that sets the level of discounting on future utility
    """
     define_objective():
     Defines the reward-maximizing objective function and computes optimal behavior using Bellman's equation and dynamic programming.
    """   
    def define_objective(self, task):    
        return self.bellmans(task)
    """
    bellmans():
    Solves Bellman's equation for reward maximization.
    """
    def bellmans(self, task):
        # Define state likelihood mesh for numerical integration:
        dp = 0.001
        p = np.arange(0, 1+dp, dp)
        # Extract reward and punishment values from probabilistic reward structure:
        R_c = task.R[0]; R_i = task.R[1] 
        # Adjust total time step budget to account for cost of commitment:
        N_mod = task.N+1-task.c[0]
        # Calculate likelihood transition between end of one decision and start of 
        # next decision given switching probability, decision direction, and reward feedback:
        p_0_pp = task.likelihood_prior_update(p, 1, 1) # Commit s_+, rewarded
        p_0_mp = task.likelihood_prior_update(p, -1, 1) # Commit s_-, rewarded
        p_0_pm = task.likelihood_prior_update(p, 1, 0) # Commit s_+, punished
        p_0_mm = task.likelihood_prior_update(p, -1, 0) # Commit s_-, punished
        # Correct for q = (0, 1) edge cases:
        if task.q == 0:
            p_0_pp[-1] = p_0_pp[0]; p_0_mm[-1] = p_0_mm[0]
            p_0_mp[0] = p_0_mp[-1]; p_0_pm[0] = p_0_pm[-1]
        elif task.q == 1:
            p_0_pp[0] = p_0_pp[-1]; p_0_mm[0] = p_0_mm[-1]
            p_0_mp[-1] = p_0_mp[0]; p_0_pm[-1] = p_0_pm[0]            
        # Round new state likelihoods to mesh and find corresponding mesh point indicies:
        sorter = np.argsort(p)
        ind_0_pp = sorter[np.searchsorted(p, np.around(p_0_pp/dp)*dp, 
                                          sorter=sorter)]
        ind_0_mp = sorter[np.searchsorted(p, np.around(p_0_mp/dp)*dp, 
                                          sorter=sorter)]
        ind_0_pm = sorter[np.searchsorted(p, np.around(p_0_pm/dp)*dp, 
                                          sorter=sorter)]
        ind_0_mm = sorter[np.searchsorted(p, np.around(p_0_mm/dp)*dp, 
                                          sorter=sorter)]
        # Construct state likelihood transfer matrix:
        F = task.likelihood_transfer(p, p)
        # Perform first step of backward induction:
        U_p = (p*(task.q*R_c+(1-task.q)*R_i)
               +(1-p)*((1-task.q)*R_c+task.q*R_i)) # Utility of choosing s_+
        U_m = ((1-p)*(task.q*R_c+(1-task.q)*R_i)
               +p*((1-task.q)*R_c+task.q*R_i)) # Utility of choosing s_-
        U_commit = np.max(np.column_stack((U_p, U_m)), axis=1) # Utility of commitment
        # Construct arguments of utility function:
        U_full = U_commit[:, None]
        # Find maximal argument of utility function and determine optimal action:
        U_max = np.zeros_like(U_commit)
        opt_act = U_max[:, None]
        # Compute utility function:
        U = U_full
        # Perform backward induction of Bellman's equation on time step budget:
        for i in range(N_mod-1, 0, -1):
            # Utility of commitment:
            if task.c[0] > N_mod-i: # If commitment exhausts time step budget
                U_p = (p*(task.q*R_c+(1-task.q)*R_i)
                       +(1-p)*((1-task.q)*R_c+task.q*R_i)) # Utility of choosing s_+
                U_m = ((1-p)*(task.q*R_c+(1-task.q)*R_i)
                       +p*((1-task.q)*R_c+task.q*R_i)) # Utility of choosing s_-
            else: # If commitment does not exhaust time step budget
                U_p = (p*(task.q*(R_c+self.gamma*U[ind_0_pp, N_mod-i-task.c[0]])
                          +(1-task.q)*(R_i+self.gamma*U[ind_0_pm, N_mod-i-task.c[0]]))
                       +(1-p)*((1-task.q)*(R_c+self.gamma*U[ind_0_pp, N_mod-i-task.c[0]])
                               +task.q*(R_i+self.gamma*U[ind_0_pm, N_mod-i-task.c[0]]))) # Utility of choosing s_+
                U_m = ((1-p)*(task.q*(R_c+self.gamma*U[ind_0_mp, N_mod-i-task.c[0]])
                              +(1-task.q)*(R_i+self.gamma*U[ind_0_mm, N_mod-i-task.c[0]]))
                       +p*((1-task.q)*(R_c+self.gamma*U[ind_0_mp, N_mod-i-task.c[0]])
                           +task.q*(R_i+self.gamma*U[ind_0_mm, N_mod-i-task.c[0]])))
            U_commit = np.max(np.column_stack((U_p, U_m)), axis=1)
            # Utility of sampling:
            if task.c[1] > N_mod-i: # If sampling exhausts time step budget
                U_sample = np.zeros_like(p)
            else: # If sampling does not exhaust time step budget
                U_sample = self.gamma*np.trapz(F*np.tile(np.transpose(U[:, N_mod-i-task.c[1]]), (len(p), 1)), p, axis=1)
            # Construct full utility function for current time step:
            U_full = np.column_stack((U_commit, U_sample))
            # Find maximal argument of utility function and determine optimal action:
            U_max = np.argmax(U_full, axis=1)
            opt_act = np.append(U_max[:, None], opt_act, axis=1)
            # Append current time step utility to total utility function:
            U = np.append(U, np.amax(U_full, axis=1)[:, None], axis=1)
        # Pad optimal action sequence to account for variable time step costs of each action:
        opt_act = np.append(opt_act, np.zeros((len(p), task.N-N_mod)), axis=1)
        return opt_act
"""
InfoMax():
Model subclass that defines the information-maximizing strategy.
"""   
class InfoMax(Model):
    def __init__(self, discounting):
        super().__init__()
        self.gamma = discounting # Parameter that sets the level of discounting on future utility
    """
    define_objective():
    Defines the information-maximizing objective function and computes optimal behavior.
    """    
    def define_objective(self, task):    
        return self.info_gain(task)
    """
    info_gain():
    Solves the information-maximization problem.
    Utilizes symmetry about state likelihood p=0.5 to simplify calculations.
    """    
    def info_gain(self, task):
        # Define state likelihood mesh for numerical integration:
        dp = 0.001
        p = np.arange(0, 1+dp, dp)
        # Adjust total time step budget to account for cost of commitment:
        N_mod = task.N+1-task.c[0]
        # Calculate likelihood transition between end of one decision and start of 
        # next decision given switching structure, decision direction, and reward feedback:
        p_0_pp = task.likelihood_prior_update(p, 1, 1) # Commit s_+, rewarded
        p_0_pm = task.likelihood_prior_update(p, 1, 0) # Commit s_+, punished
        # Correct for q = (0, 1) edge cases:
        if task.q == 0:
            p_0_pp[-1] = p_0_pp[0]
            p_0_pm[0] = p_0_pm[-1]
        elif task.q == 1:
            p_0_pp[0] = p_0_pp[-1]
            p_0_pm[-1] = p_0_pm[0]   
        # Round new state likelihoods to mesh and find corresponding mesh point indicies:
        sorter = np.argsort(p)
        ind_0_pp = sorter[np.searchsorted(p, np.around(p_0_pp/dp)*dp, 
                                          sorter=sorter)]
        ind_0_pm = sorter[np.searchsorted(p, np.around(p_0_pm/dp)*dp, 
                                          sorter=sorter)]
        # Construct state likelihood transfer matrix:
        F = task.likelihood_transfer(p, p)
        # Perform first step of backward induction:
        U_commit = np.zeros_like(p) # Utility of commitment
        # Construct arguments of utility function:
        U_full = U_commit[:, None]
        # Find maximal argument of utility function and determine optimal action:
        U_max = np.zeros_like(U_full)
        opt_act = U_max
        # Compute utility function:
        U = U_full
        # Perform backward induction of Bellman's equation on time step budget:
        for i in range(N_mod-1, 0, -1):
            # Utility of commitment:
            if task.c[0] > N_mod-i: # If commitment exhausts time step budget
                U_commit = np.zeros_like(p)
            else: # If commitment does not exhaust time step budget
                U_commit = (self.likelihood_state_entropy(p+task.epsilon-2*p*task.epsilon)-self.likelihood_commit_entropy(p, task)
                             +p*(task.q*self.gamma*U[ind_0_pp, N_mod-i-task.c[0]]
                                 +(1-task.q)*self.gamma*U[ind_0_pm, N_mod-i-task.c[0]])
                             +(1-p)*((1-task.q)*self.gamma*U[ind_0_pp, N_mod-i-task.c[0]]
                                     +task.q*self.gamma*U[ind_0_pm, N_mod-i-task.c[0]]))
            # Utility of sampling:
            if task.c[1] > N_mod-i: # If sampling exhausts time step budget
                U_sample = -np.inf*np.ones_like(p)
            else: # If sampling does not exhaust time step budget
                U_sample = (self.likelihood_state_entropy(p)-self.likelihood_sample_entropy(p, task)
                             +self.gamma*np.trapz(F*np.tile(np.transpose(U[:, N_mod-i-task.c[1]]), (len(p), 1)), p, axis=1))
            # Construct full utility function for current time step:
            U_full = np.round(np.column_stack((U_commit, U_sample)), decimals=15)
            # Find maximal argument of utility function and determine optimal action:
            U_max = np.argmax(U_full, axis=1)
            opt_act = np.append(U_max[:, None], opt_act, axis=1)
            # Append current time step utility to total utility function:
            U = np.append(U, np.amax(U_full, axis=1)[:, None], axis=1)
        # Pad optimal action sequence to account for variable time step costs of each action:
        opt_act = np.append(opt_act, np.zeros((len(p), task.N-N_mod)), axis=1)
        return opt_act
    """
    likelihood_state_entropy():
    Calculates the uncertainty about the current environmental state based on the agent's state likelihood belief.
    """
    def likelihood_state_entropy(self, p):
        entropy = -p*np.log2(p)-(1-p)*np.log2(1-p)
        # Correct for state belief value edge cases p=(0, 1):
        entropy[p == 0] = 0; entropy[p == 1] = 0   
        return entropy
    """
    likelihood_commit_entropy():
    Calculates the expected uncertainty on next time step if the agent commits.
    """
    def likelihood_commit_entropy(self, p, task):
        # Calculate for task parameter edge cases:
        if (((task.epsilon == 0) | (task.epsilon == 1)) 
            & ((task.q == 0) | (task.q == 1))):
            entropy = 0
        else:
            pp = task.likelihood_prior_update(p, 1, 1)
            pm = task.likelihood_prior_update(p, 1, 0)
            # Correct for reward reliability q=(0, 1) edge cases:
            if task.q == 0:
                pp[-1] = pp[0]
                pm[0] = pm[-1]
            elif task.q == 1:
                pp[0] = pp[-1]
                pm[-1] = pm[0]
            entropy = (task.q*p+(1-task.q)*(1-p))*self.likelihood_state_entropy(pp)
            entropy += ((1-task.q)*p+task.q*(1-p))*self.likelihood_state_entropy(pm)    
        return entropy
    """
    likelihood_sample_entropy():
    Calculates the expected uncertainty on the next time step if the agent samples.
    """
    def likelihood_sample_entropy(self, p, task):
        # Construct likelihood transfer matrix:
        F = task.likelihood_transfer(p, p)
        entropy = np.trapz(F*np.tile(self.likelihood_state_entropy(p), (len(p), 1)), p, axis=1)
        return entropy
