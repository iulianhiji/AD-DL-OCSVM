# Fusing Dictionary Learning and Support Vector Machines for Unsupervised Anomaly Detection

Implementation of the Fused DL and Support Vector Machines for AD algorithm described in



The algorithms are implemented in [ksvd_supp.py](ksvd_supp.py). Have a look at the experiments for full examples:
* [DL-OCSVM](code/test_DL_OCSVM.py)
* [DPL-OCSVM](code/test_DPL_OCSVM.py) 
* [KDL-OCSVM](code/test_KDL_OCSVM.py) 
* [KDPL-OCSVM](code/test_KDPL_OCSVM.py) 

To generate the random dictionaries for all datasets, call the script [generate_dicts](code/generate_dicts.py) 
All the scripts must be called from [code](code) directory


## Requirements

* the [graphomaly package](https://gitlab.com/unibuc/graphomaly/graphomaly)
* attridict
* numpy==1.22
* cvxpy
```
pip install graphomaly
pip install attridict
pip install numpy==1.22
pip install cvxpy


```

* the [ODDS database(mat files -- already in /data)](http://odds.cs.stonybrook.edu/) for the real-data experiments
