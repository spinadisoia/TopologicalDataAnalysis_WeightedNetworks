# TopologicalDataAnalysis_WeightedNetworks
Using topological tools to enrich our understanding on weighted complex networks. A case study and a framework for applying persistent homology on weighted networks


This repository contains the implementation and analysis tools for studying the topological properties of weighted complex networks using methods from persistent homology. The project addresses the challenge of analyzing the higher-order organization of weighted networks, which traditional graph-theoretic approaches often fail to capture.

This project aims to study the structure of weighted complex networks preserving its full complexity by introducing a method to:
-  Use filtration to take into account the weights
-  Characterize the evolution of the network by studying th "holes", which represent low connectivity regions
-  Classify networks based on their topological similarity to randomized null models.

### Codes
-  auxiliary_functions.py:
    contains utility functions for visualizing weighted networks, computing persistent homology, and analyzing topological features (e.g., filtrations, persistence diagrams, hollowness indexes).
-  toy_network.ipynb:
    example network for the presentation.
-  Class1_network.ipynb
    case study of a simple network simulating the behaviour of class 1 networks.
-  Class2_network.ipynb:
    case study of a simple network simulating the behaviour of class 2 networks.
-  vizualization_paper.ipynb:
    barplots visualizing the results of the papers.

### References 
Petri et al. (2013). Topological Strata of Weighted Complex Networks. PLoS ONE.
Petri et al. (2014). Homological scaffolds of brain functional networks. J. R. Soc. Interface.
