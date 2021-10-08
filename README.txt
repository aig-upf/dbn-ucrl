
Python code for running experiments with different versions of UCRL and Factored UCRL.

The experiments are defined in RL_FMDP_XXX.py, where XXX is the name of the domain.

The rest of the code is organized as follows:

- MDP.py represents the true model of an MDP, while FactoredMDP.py represents the true model of an FMDP. Both are subclasses of AbstractMDP.

- The domain used in experiments are all subclasses of FactoredMDP.

- AbstractUCRL.py contains the main loop of UCRL, but defers details (updatepolicy and updateparams) to its subclasses.

- UCRL.py implements UCRL for a standard MDP, while FactoredUCRL.py implements UCRL for FMDPs.

- Confidence intervals for both rewards and transition probabilities are computed using classes in ConfidenceBounds.py.

- Different versions of EVI are implemented using classes in ExtendedValueIteration.py.

The code is based on https://github.com/aig-upf/dbn-ucrl.
