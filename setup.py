import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = []
    for library in f.read().splitlines():
        requirements.append(library)

setuptools.setup(
    name="clustviz",  # Replace with your own username
    version="0.0.1",
    author="Guglielmo Sanchini",
    author_email="guglielmosanchini@gmail.com",
    description="A 2D clustering algorithms visualization package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/guglielmosanchini/ClustViz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements
)
