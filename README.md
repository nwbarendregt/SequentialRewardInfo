Python code used to generate all figures for Barendregt et. al., "Information-Seeking Decision Strategies Mitigate Risk in Dynamic, Uncertain Environments."
# Manuscript Link
The preprint of our manuscript can be found here: [https://arxiv.org/abs/2503.19107](https://arxiv.org/abs/2503.19107)

# Repository Structure
The directory [sequential_methods](/sequential_methods) is a Python module that contains all the methods and class definitions necessary to construct the sequential task proposed in our work, define the rewardmax and infomax decision strategies, and simulate their behavior on the given task. 
The module was created using abstract base classes so that other task designs and decision strategies can be incorporated as extensions of our work. 
The directory [Task_Simulation](/Task_Simulation) contains example scrips that generate individual realizations of rewardmax and infomax behavior, as well as parameter sweep scripts that extract performance and behavioral statistics across a range of environmental parameterizations.

To generate the figures from the manuscript, each figure X has a directory "Figure_X" that contains both the Python script "figure_X_generate.py" and all csv data files necessary to generate the figure. If you are interested in replicating the figures from the manuscript using our data sets, run the Python script contained in the desired figure's directory. 
If you are interested in generating new data sets, we recommend using the scripts in the [Task_Simulation](/Task_Simulation) directory to generate new csv files.
