# CLUSTERING VISUALIZATION
Visualizing clustering algorithms step by step

The aim of this project is to visualize every step of each clustering algorithm, in the case of 2D input data.

The following algorithms have been examined:
- OPTICS
- DBSCAN
- HDBSCAN
- SPECTRAL CLUSTERING
- HIERARCHICAL AGGLOMERATIVE CLUSTERING
  - single linkage
  - complete linkage
  - average linkage
  - Ward's method
- CURE
- BIRCH
- PAM
- CLARA
- CLARANS
- CHAMELEON
- CHAMELEON2
- DENCLUE


## Repository structure

1) The folder **DOCUMENTS** contains all the official papers, powerpoint presentations and other PDFs regarding all the algorithms involved and clustering in general

2) The folder **algorithms** contains the scripts necessary to run the algorithms

3) The folder **metis-5.1.0** contains the Metis library (https://metis.readthedocs.io/en/latest/)

4) The notebook **Clustering_visualization_notebook** lets the user run every algorithm on 2D datasets; it contains a subsection for every algorithm, with the necessary modules and functions imported and some commented lines of code which can be uncommented to run the algorithms


## Credits for some algorithms
I did not start to write the scripts for each algorithm from scratch; in some cases I modified some Python libraries, in other cases I took some publicly available GitHub repositories and modified the scripts contained there. The following list provides all the sources used when I did not write all the code by myself:

- HDBSCAN
https://hdbscan.readthedocs.io/en/latest/
- SPECTRAL CLUSTERING
http://dx.doi.org/10.1007/s11222-007-9033-z
- BIRCH
https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/birch.py
