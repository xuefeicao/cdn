# Causal Dynamic Network Modeling
cdn-fmri is a Python-based package implementing causal dynamic network analysis for Functional magnetic resonance imaging (fMRI). It aims to improve the dynamic causal modeling with the optimization based method. Our optimization-based method and algorithm compute efficiently the ODE parameters from fMRI data, instead of comparing potentially a huge number of candidate ODE models. For a detailed introduction of dynamic causal modeling, see [(1)].



## Getting Started
This package supports both python 2.7 and python 3.6.

Example provided in the repo has been tested in mac os and Linux environment. 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

This package is also published in pypi.  For a quick installation, try

```
pip install cdn-fmri 
```

### Prerequisites

What things you need to install the software and how to install them

```
See setup.py for details of packages requirements. 
```

### Installing from GitHub


Download the packages by using git clone https://github.com/xuefeicao/cdn.git

```
python setup.py install
```

If you experience problems related to installing the dependency Matplotlib on OSX, please see https://matplotlib.org/faq/osx_framework.html 

### Intro to our package
After installing our package locally, try to import cdn in your python environment and learn about package's function. 
Note: our package name in pypi is cdn-fmri.
```
import cdn.cdn_analysis as cdn_analysis
help(cdn_analysis.cdn_multi_sub)
```


### Examples
```
The examples subfolder includes a basic analysis of CDN analysis.
```

## Built With

* Python 2.7

## Compatibility
* python 2.7
* python 3.6 

## Authors

* **Xuefei Cao** - *Maintainer* - (https://github.com/xuefeicao)
* **Xi Luo** (http://bigcomplexdata.com/)
* **Björn Sandstede** (http://www.dam.brown.edu/people/sandsted/)


## License

This project is licensed under the MIT License - see the LICENSE file for details

[(1)]:http://www.fil.ion.ucl.ac.uk/~karl/Dynamic%20causal%20modelling.pdf
