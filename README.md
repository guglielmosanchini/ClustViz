[![Build Status](https://travis-ci.com/guglielmosanchini/ClustViz.svg?branch=master)](https://travis-ci.com/guglielmosanchini/ClustViz)
[![codecov](https://codecov.io/gh/guglielmosanchini/ClustViz/branch/master/graph/badge.svg)](https://codecov.io/gh/guglielmosanchini/ClustViz)
[![Documentation Status](https://readthedocs.org/projects/clustviz/badge/?version=latest)](https://clustviz.readthedocs.io/en/latest/?badge=latest)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# ClustViz
<img src="https://raw.githubusercontent.com/guglielmosanchini/ClustViz/master/data/clustviz_logo.png" width="200" height="200">

## 2D Clustering Algorithms Visualization

#### Check out [ClustVizGUI](https://github.com/guglielmosanchini/ClustVizGUI), too!
The aim of ```ClustViz``` is to visualize every step of each clustering algorithm, in the case of 2D input data.

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

**Documentation**: [click here](https://clustviz.readthedocs.io/en/latest/)

Install with 
```python
pip install clustviz
```

To run BIRCH algorithm, the open source visualization software Graphviz is required. 
Install Graphviz from the official webpage (https://graphviz.gitlab.io/download/) or using HomeBrew, then 
modify the PATH variable as follows (replace the string according to the path where you installed Graphviz):

```python
import os
# on Windows usually
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
# on MacOS usually
os.environ["PATH"] += os.pathsep + '/usr/local/bin'
```

To run CHAMELEON and CHAMELEON2 algorithms, the [METIS](https://metis.readthedocs.io/en/latest/) library is required.
To install it on MacOS, execute the following commands (partially taken from [here](http://glaros.dtc.umn.edu/gkhome/metis/metis/download)):

```bash
# download the file using wget (do it from the website if you prefer)
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz
# uncompress it
gunzip metis-5.1.0.tar.gz
# untar it
tar -xvf metis-5.1.0.tar
# remove the tar
rm metis-5.1.0.tar
# go inside the folder
cd metis-5.1.0
# install it using make
make config shared=1
make install
# export the dll
export METIS_DLL=/usr/local/lib/libmetis.dylib
```

To install METIS on Windows, go to [conda-metis](https://github.com/guglielmosanchini/conda-metis) and follow the instructions.

## Usage
Let's see a basic example using OPTICS:

```python
from clustviz.optics import OPTICS, plot_clust
from sklearn.datasets import make_blobs

# create a random dataset
X, y = make_blobs(n_samples=30, centers=4, n_features=2, cluster_std=1.8, random_state=42)

# perform OPTICS algorithm, with plotting enabled
ClustDist, CoreDist = OPTICS(X, eps=2, minPTS=3, plot=True, plot_reach=True)

# plot the final clusters
plot_clust(X, ClustDist, CoreDist, eps=2, eps_db=1.9)
```

For many other examples, take a look at the detailed [clustviz_example](https://github.com/guglielmosanchini/ClustViz/blob/master/data/clustviz_example.ipynb) notebook.

## Repository structure

1) The folder ```data/DOCUMENTS``` contains all the official papers, powerpoint presentations and other PDFs regarding all the algorithms involved and clustering in general.

2) The folder ```clustviz``` contains the scripts necessary to run the clustering algorithms.

3) The notebook ```data/clustviz_example.ipynb``` lets the user run every algorithm on 2D datasets; it contains a subsection for every algorithm, with the necessary modules and functions imported and some commented lines of code which can be uncommented to run the algorithms.

4) The folder ```docs``` contains the necessary files to build the documentation using Sphinx and ReadTheDocs.

5) The folder ```tests``` contains pytest tests.

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
the GUI Meister, for the help with PyQt5, used for [ClustVizGUI](https://github.com/guglielmosanchini/ClustVizGUI).

## Possible improvements
- add more clustering algorithms
- comment every code block and improve code quality
- pymetis doesnt work on Windows, but could be an option for MacOS
- add highlights to docstrings using ``
- show aliases typehints using Sphinx (open issue)

## TravisCI path
- if Travis CI doesn't trigger, it is probably because ```.travis.yml``` isn't properly formatted. Use
```yamllint``` to correct it
- add package update
- for the deployment phase: brew install ruby, brew install travis
- added empty conftest.py in clustviz folder for tests in windows version
