Sphinx pipeline

cd docs

- make clean (to delete old html)
- delete clustviz.rst and modules.rst in docs/source 
- sphinx-apidoc -f -o source ../clustviz
- make html

https://medium.com/better-programming/auto-documenting-a-python-project-using-sphinx-8878f9ddc6e9