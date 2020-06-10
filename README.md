[![Build Status](https://travis-ci.com/guglielmosanchini/Clustering.svg?branch=master)](https://travis-ci.com/guglielmosanchini/Clustering)
[![Coverage Status](https://coveralls.io/repos/github/guglielmosanchini/Clustering/badge.svg?branch=master)](https://coveralls.io/github/guglielmosanchini/Clustering?branch=master)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

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

## Instructions
Just open the notebook **Clustering_visualization_notebook** and run whatever section you like, using 2D datasets (for visualization purposes, the cardinality of the datasets should be <= 250 points), to see each algorithm in action.

Alternatively, run the script **gui.py** to open a GUI built with PyQt5 and use it to explore the clustering algorithms; currently, Chameleon and Chameleon2 are
only supported on MacOS, and may be impossible to use in a Windows environment due to difficulties in installing the METIS library (the way to use **make** in Windows explained [here](https://stackoverflow.com/questions/32127524/how-to-install-and-use-make-in-windows) could solve the problem).

To run BIRCH algorithm, the open source visualization software Graphviz is required. To install it, write in the command line
```python
pip install graphviz
```
and then, after installing Graphviz from the official webpage (https://graphviz.gitlab.io/download/) or using HomeBrew,
the PATH variable has to be modified as follows (replace the string according to the path where you installed Graphviz):
```python
import os
# on Windows usually
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
# on MacOS usually
os.environ["PATH"] += os.pathsep + '/usr/local/bin'
```

## Repository structure

1) The folder **DOCUMENTS** contains all the official papers, powerpoint presentations and other PDFs regarding all the algorithms involved and clustering in general.

2) The folder **algorithms** contains the scripts necessary to run the algorithms.

3) The folder **GUI_classes** contains the scripts necessary to run the GUI.

4) The folder **Images** serves the purpose of storing the images plotted when using the GUI, if desired by the user.

5) The notebook **Clustering_visualization_notebook** lets the user run every algorithm on 2D datasets; it contains a subsection for every algorithm, with the necessary modules and functions imported and some commented lines of code which can be uncommented to run the algorithms.

6) The script **Clustering_visualization_notebook** is just a .py version of the notebook.

7) The script **gui** starts the GUI for the visualization of clustering algorithms.

<img src="https://raw.githubusercontent.com/guglielmosanchini/Clustering/master/clustviz/Images/README_pics/pic1_gui.JPG" width="450" height="350">

<img src="https://raw.githubusercontent.com/guglielmosanchini/Clustering/master/clustviz/Images/README_pics/pic2_gui.JPG" width="450" height="350">

<img src="https://raw.githubusercontent.com/guglielmosanchini/Clustering/master/clustviz/Images/README_pics/pic3_gui.JPG" width="450" height="350">

## Credits for some algorithms
I did not start to write the scripts for each algorithm from scratch; in some cases I modified some Python libraries, in other cases I took some publicly available GitHub repositories and modified the scripts contained there. The following list provides all the sources used when I did not write all the code by myself:

- HDBSCAN
https://hdbscan.readthedocs.io/en/latest/
- SPECTRAL CLUSTERING
http://dx.doi.org/10.1007/s11222-007-9033-z
- BIRCH
https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/birch.py
- PAM
https://github.com/SachinKalsi/kmedoids/blob/master/KMedoids.py
- CLARA
https://github.com/akalino/Clustering/blob/master/clara.py
- CLARANS
https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/clarans.py
- CHAMELEON
https://github.com/Moonpuck/chameleon_cluster

The other algorithms have been implemented from scratch following the relative papers. Thanks to Darius (https://github.com/dariomonici), 
the GUI Meister, for the help with PyQt5.

## Possible improvements
- wrap it as a python package
- add more clustering algorithms
- add a pause/resume button for every algorithm
- comment every code block and improve code quality

## TravisCI path
- added empty ```conftest.py``` in ```clustviz``` to make **pytest** work, otherwise it wasn't able to import
any of the modules inside ```clustviz```.
- if Travis CI doesn't trigger, it is probably because ```.travis.yml``` isn't properly formatted. Use
```yamllint``` to correct it
- add package update
