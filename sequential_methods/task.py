"""
task.py
Contains class object for task design and parameterization.
Specific task designs are defined as subclasses of the Task superclass.
"""
import numpy as np
rng = np.random.default_rng() # Creates numpy random number generator for stochastic simulations
from abc import ABC, abstractmethod # Used for creating Task superclass
"""
Task():
Superclass that dictates the structural requirements of each task subclass.
Also contains functions common to all task subclasses.
"""
class Task(ABC):
    def __init__(self, N, epsilon, q, R, c):
        self.N = N # total time step budget
        self.epsilon = epsilon # state switching probability
        self.q = q # reward feedback reliability
        self.R = R # reward structure tuple (reward, punishment)
        self.c = c # action time step cost tuple (commit cost, sample cost)
    """
    belief_sim():
    Simulates belief evolution for individual decisions (defined in each task subclass).
    """    
    @abstractmethod
    def belief_sim(self):
        raise NotImplementedError()
    """
    likelihood_transfer():
    Calculates the probability transfer function for future belief given current belief (defined in each task subclass).
    """    
    @abstractmethod
    def likelihood_transfer(self):
        raise NotImplementedError()
    """
    prior_update():
    Calculates belief evolution from the end of one decision to the begining of the next decision based on switching rate, decision direction, and reward reliability.
    Returns prior in units of LLR belief.
    """    
    def prior_update(self, y_d, d, r):
        if r == 1: # If agent is rewarded
            # Correct for parameter edge cases:
            if ((self.epsilon == 0) & (self.q == 1)) | ((self.epsilon == 1) & (self.q == 0)):
                y_0 = d*np.inf
            elif ((self.epsilon == 0) & (self.q == 0)) | ((self.epsilon == 1) & (self.q == 1)):
                y_0 = d*(-np.inf)
            else:
                y_0 = d*np.log(((1-self.epsilon)*self.q*np.exp(np.abs(y_d))+self.epsilon*(1-self.q))
                                      /(self.epsilon*self.q*np.exp(np.abs(y_d))+(1-self.epsilon)*(1-self.q)))
        else: # if agent is punished
            # Correct for parameter edge cases:
            if ((self.epsilon == 0) & (self.q == 0)) | ((self.epsilon == 1) & (self.q == 1)):
                y_0 = d*np.inf
            elif ((self.epsilon == 0) & (self.q == 1)) | ((self.epsilon == 1) & (self.q == 0)):
                y_0 = d*(-np.inf)
            else:
                y_0 = d*np.log(((1-self.epsilon)*(1-self.q)*np.exp(np.abs(y_d))+self.epsilon*self.q)
                                      /(self.epsilon*(1-self.q)*np.exp(np.abs(y_d))+(1-self.epsilon)*self.q))
        return y_0
    """
    prior_update():
    Calculates belief evolution from the end of one decision to the begining of the next decision based on switching rate, decision direction, and reward reliability.
    Returns prior in units of state likelihood belief.
    """    
    def likelihood_prior_update(self, p_d, d, r):
        if ((r == 1) & (d == 1)) | ((r == 0) & (d == -1)):
            p_0 = (((1-self.epsilon)*self.q*p_d+self.epsilon*(1-self.q)*(1-p_d))
                   /((1-self.epsilon)*self.q*p_d+self.epsilon*(1-self.q)*(1-p_d)
                     +self.epsilon*self.q*p_d+(1-self.epsilon)*(1-self.q)*(1-p_d)))
        if ((r == 1) & (d == -1)) | ((r == 0) & (d == 1)):
            p_0 = (((1-self.epsilon)*(1-self.q)*p_d+self.epsilon*self.q*(1-p_d))
                   /((1-self.epsilon)*(1-self.q)*p_d+self.epsilon*self.q*(1-p_d)
                     +self.epsilon*(1-self.q)*p_d+(1-self.epsilon)*self.q*(1-p_d)))
        return p_0            
"""
BernoulliInference():
Task subclass that defines the environmental evidence as Bernoulli-distributed.
"""
class BernoulliInference(Task):
    def __init__(self, N, epsilon, q, R, c, h):
        super().__init__(N, epsilon, q, R, c) # Inherit properties of super class
        self.h = h # parameter of Bernoulli-evidence process
    """
    belief_sim():
    Simulates LLR belief trajectory for current decision using Bernoulli-generated environmental evidence.
    """
    def belief_sim(self, y_0, s):
        # Determine Bernoulli parameter based on current state:
        h_s = self.h*(s == +1)+(1-self.h)*(s == -1)
        # Pre-allocate belief trajectory vector and initialize with prior belief:
        y = np.empty(self.N); y[:] = np.nan; y[0] = y_0
        for i in range(self.c[1], self.N, self.c[1]):
            xi = 2*rng.binomial(1, h_s)-1 # Draw observation
            y[i] = xi*np.log(self.h/(1-self.h))+y[i-self.c[1]] # Update LLR
        return y
    """
    likelihood_transfer():
    Construct likelihood transfer function using Bernoulli-generated environmental evidence.
    """
    def likelihood_transfer(self, p, p0):
        # Construct state likelihood mesh and pre-allocate transfer matrix:
        dp = 0.001
        F = np.zeros((len(p), len(p0)))
        for i in range(len(p0)):
            # Compute possible new state likelihood values based on current state likelihood p0
            pp = self.h*p0[i]/((1-self.h)*(1-p0[i])+self.h*p0[i]) 
            pm = (1-self.h)*p0[i]/(self.h*(1-p0[i])+(1-self.h)*p0[i])
            # Find new state likelihood value indicies in state likelihood mesh:
            pp_ind = np.argwhere(p == np.round(pp/dp)*dp)
            pm_ind = np.argwhere(p == np.round(pm/dp)*dp)
            # Store state likelihood transfer probabilities in appropriate mesh locations:
            F[i, pp_ind] = (p0[i]*self.h+(1-p0[i])*(1-self.h))/dp
            F[i, pm_ind] = (p0[i]*(1-self.h)+(1-p0[i])*self.h)/dp
        # Correct transfer matrix for state likelihoods of 0:
        if p0[0] == 0:
            F[0, :] = 0
        if p[0] == 0:
            F[:, 0] = 0
        if (p0[0] == 0) & (p[0] == 0):
            F[0, 0] = 2/dp
        # Correct transfer matrix for state likelihoods of 1:
        if p0[-1] == 1:
            F[-1, :] = 0
        if p[-1] == 1:
            F[:, -1] = 0
        if (p0[-1] == 1) & (p[-1] == 1):
            F[-1, -1] = 2/dp
        return F
