# PairwiseApproach

### Experiment: Accuracy of sign(Y_c2)

Run /code/run_Yc2_accuracy.py for all the QSAR datasets (up to the number of datasets specified in the manuscript).


### Experiment: Extrapolation on ChEMBL datasets

Run /code/run_extrapolation.py for all the QSAR datasets (up to the number of datasets specified in the manuscript).


The default ML method used in this version of code is random forests. To get results for other ML methods (with parallelisation enabled in sci-kit learn), you will need to manually change RandomForestRegressor and RandomForestClassifier into other methods in build_model_xxx.py.

To enable reproducibility, random_state=1 is used in the code whenever a method requires this as an input.


