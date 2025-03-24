"""
experiment.py
Combines task and model classes to simulate behavior in an experiment. 
Generates dataclass to store results of experimental simulation for further analysis.
"""
import numpy as np
rng = np.random.default_rng() # Creates numpy random number generator for stochastic simulations
np.seterr(divide='ignore', invalid='ignore') # Supresses warnings from entropy calculations
from dataclasses import dataclass, field
from typing import Tuple

"""
Behavior():
Class object that combines task and model classes that are used to simulate behavior in an experiment.
"""
class Behavior():
    
    def __init__(self, task, model):
        self.task = task # Task class obtained from task.py that defines task structure and parameters
        self.model = model # Model class obtained from model.py that defines agent's strategy
        self.opt_action = self.model.define_objective(self.task) # Construct decision strategy using agent's model
"""
Results():
Dataclass for storing results of experiment simulation for further analysis.
"""
@dataclass
class Results():
     
    choice: np.ndarray = field(default_factory = lambda: np.array([])) # vector of state choices
    RT: np.ndarray = field(default_factory = lambda: np.array([])) # vector of choice times
    correct_state: np.ndarray = field(default_factory = lambda: np.array([])) # vector of correct states
    reward: np.ndarray = field(default_factory = lambda: np.array([])) # vector of obtained rewards
    belief: np.ndarray = field(default_factory = lambda: np.array([])) # vector of agent's belief values
    action: list = field(default_factory = lambda: []) # vector of agent's actions
    
    """
    add_decision():
    Adds data from the current decision to the dataclass.
    """
    def add_decision(self, RT, choice, correct_state, reward, belief):
        self.RT = np.append(self.RT, RT)
        self.choice = np.append(self.choice, choice)
        self.correct_state = np.append(self.correct_state, correct_state)
        self.reward = np.append(self.reward, reward)
        self.belief = np.append(self.belief, belief)
        self.action.append('s'*(np.sum(~np.isnan(belief))-1) + 'c')
    """
    add_metadata():
    Adds task structure metadata for data classification and sorting.
    """
    def add_metadata(self, metadata):
        self.metadata = metadata
"""
Metadata():
Dataclass for storing relevant metadata about experimental simulation.
"""
@dataclass
class MetaData():
    
    N: int # Time step budget
    epsilon: float # Environmental switching probability
    q: float # Reward/punishment reliability
    R: Tuple # Reward structure values in the form of the tuple (reward, punishment)
    c: Tuple # Time step costs of each action in the form of (commit cost, sample cost)

"""
simulate():
Simulates agent's behavior in the sequential task over a specified time step budget.
"""
def simulate(behavior):
# Initialize dataclass for storing results and metadata of experiment:
    results = Results()
    # Define initial belief and environmental state:
    y_0 = 0
    s_i = 2*rng.binomial(1, 0.5)-1
    # Track number of time steps spent:
    n = behavior.task.N
    while n >= behavior.task.c[0]:
        # Simulate normative belief for current environmental state:
        y_i = behavior.task.belief_sim(y_0, s_i)
        # Calculate response time and choice on current environmental state:
        response = decision_sim(y_i, n, behavior.task, behavior.opt_action)
        # Determine reward/punishment feedback on current decision:
        feedback = rng.binomial(1, 
                                (behavior.task.q*(s_i != response[1])+(1-behavior.task.q)*(s_i == response[1])))
        # Store statistics from decision:
        results.add_decision(response[0], 
                      response[1], 
                      s_i, 
                      behavior.task.R[feedback], 
                      np.append(y_i[:n-response[0]+1], 
                                np.array([np.nan]*(behavior.task.c[0]-1))))
        # Update time steps spent counter:
        n = response[0]-behavior.task.c[0]
        # Update initial belief and correct choice for next environmental state if budget is not exhausted:
        if n >= behavior.task.c[0]:
            y_d = results.belief[-behavior.task.c[0]]
            y_0 = behavior.task.prior_update(y_d, response[1], 1-feedback)
            swap = rng.binomial(1, behavior.task.epsilon) # Swap environmental state probabilistically
            s_i = s_i*(swap == 0)-s_i*(swap == 1)
    # Pad belief trajectory if final commitment costs more than one time step:    
    if len(results.belief) < behavior.task.N:
        results.belief = np.append(results.belief, 
                                   np.array([np.nan]*(behavior.task.N-len(results.belief))))
    # Store experimental metadata:    
    results.add_metadata(MetaData(behavior.task.N, 
                              behavior.task.epsilon, 
                              behavior.task.q, 
                              behavior.task.R, 
                              behavior.task.c))        
    return results
"""
decision_sim():
Simulates agent's choice and response time based on their decision strategy and belief trajectory.
"""    
def decision_sim(y, n, task, opt_action):
    # Generate state likelihood mesh:
    p_mesh = np.linspace(0, 1, len(opt_action[:, 0]))
    # Initialize response time, decision, and agent's decision strategy:
    RT = 0; choice = np.NaN
    opt_act_n = opt_action[:, task.N-n:]
    while np.isnan(choice):
        # Convert LLR belief to state likelihood belief:
        p = 1/(1+np.exp(-y[RT]))
        # Determine if agent will sample or commit based on their strategy and current belief:
        if opt_act_n[np.argmin(np.abs(p-p_mesh)), RT] == 1:
            RT += task.c[1]
        else:
            choice = np.sign(y[RT])
    # If agent commits with LLR belief of 0, set response randomly with equal probability given to each possible state:            
    if choice == 0:
        choice = 2*np.random.binomial(1, 0.5)-1
    RT = n-RT
    return RT, choice
