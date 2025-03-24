"""
__init__.py
init file that loads each module used for simulating sequential decision task.
"""

from . import task # Contains classes for generating task structure
from . import model # Contains classes for constructing decision models
from . import experiment # Combines task and model classes to simulate experiment
from . import analysis # Performs statistical analysis of experimental simulations
