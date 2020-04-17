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
only supported on Mac OS, and may be impossible to use in a Windows environment due to difficulties in installing the METIS library (the way to use **make** in Windows explained [here](https://stackoverflow.com/questions/32127524/how-to-install-and-use-make-in-windows) could solve the problem).

To run BIRCH algorithm, the open source visualization software Graphviz is required. To install it, write in the command line
```python
pip install graphviz
```
and then, after installing Graphviz from the official webpage (https://graphviz.gitlab.io/download/) or using HomeBrew,
the PATH variable has to be modified as follows (replace the string according to the path where you installed Graphviz):
```python
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
```

To run Chameleon and Chameleon2 algorithm the library METIS is required. To install it on Mac OS, execute the following steps:
```python
pip install metis
```
Then, open the terminal, go to the **metis-5.1.0** folder and run
```python
make config shared=1
```
and
```python
make install
```
## Repository structure

1) The folder **DOCUMENTS** contains all the official papers, powerpoint presentations and other PDFs regarding all the algorithms involved and clustering in general.

2) The folder **algorithms** contains the scripts necessary to run the algorithms.

3) The folder **GUI_classes** contains the scripts necessary to run the GUI.

4) The folder **Images** serves the purpose of storing the images plotted when using the GUI, if desired by the user.

5) The folder **metis-5.1.0** contains the Metis library (https://metis.readthedocs.io/en/latest/).

6) The notebook **Clustering_visualization_notebook** lets the user run every algorithm on 2D datasets; it contains a subsection for every algorithm, with the necessary modules and functions imported and some commented lines of code which can be uncommented to run the algorithms.

7) The script **Clustering_visualization_notebook** is just a .py version of the notebook.

8) The script **gui** starts the GUI for the visualization of clustering algorithms.

<img src="https://raw.githubusercontent.com/guglielmosanchini/Clustering/master/Images/README_pics/pic1_gui.JPG" width="450" height="350">

<img src="https://raw.githubusercontent.com/guglielmosanchini/Clustering/master/Images/README_pics/pic2_gui.JPG" width="450" height="350">

<img src="https://raw.githubusercontent.com/guglielmosanchini/Clustering/master/Images/README_pics/pic3_gui.JPG" width="450" height="350">

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
