Sphinx pipeline

```cd docs```

- ```pip install sphinx```
- ```pip install sphinx_rtd_theme``` # or any other theme you wish to use
- ```pip install sphinx-autodoc-typehints```
- ```make clean``` (to delete old html)
- delete ```clustviz.rst``` and ```modules.rst``` in ```docs/source ```
- ```sphinx-apidoc -f -o source ../clustviz```
- packages whose names contain underscore(s) are ignored by sphinx, so two files need to 
be created by hand: ```clustviz._birch.rst``` and ```clustviz._chameleon.rst```; then, ```clustviz.rst``` must 
be modified by hand to include ```clustviz._birch``` and ```clustviz._chameleon``` as subpackages
- ```make html```




https://medium.com/better-programming/auto-documenting-a-python-project-using-sphinx-8878f9ddc6e9