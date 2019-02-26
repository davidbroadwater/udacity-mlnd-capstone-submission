# Gradient Boosted Trees for the Prediction of Professional Tennis Matches
## Capstone project for Udacity Machine Learning Nanodegree

### Packages/Software

All code was written in a Jupyter notebook (`MLND Capstone.ipynb`) running Python 3.7. A list of the libraries and packages is provided below, along with links for the less well known packages. All packages listed can be installed via pip. Note that n_jobs has been set to `-1` (max number available) in most cases. On my 2012 4-core iMac, the entire notebook takes a couple hours to run (primarily due to two ~40 minute parameter tuning steps; YMMV).

Packages:
- pandas
- numpy
- matplotlib
- sklearn
- time
- [pandas_profiling](https://github.com/pandas-profiling/pandas-profiling)
- [xgboost](https://xgboost.readthedocs.io/en/latest/)
- [skopt](https://scikit-optimize.github.io)


### Data
All of the data required for this project is located in `ATP.csv`. 

### Data Profiles
Data profiles are generated at various stages of the data processing pipeline as part of the notebook, and can be opened in a browser. 

