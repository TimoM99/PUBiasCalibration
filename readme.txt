This folder contains code to run the methods discussed in 'Learning from biased positive-unlabeled data via threshold calibration'

The folder Models contains the different methods discussed in the experimental section. Underlying they use the code in the folders lbe (written by us based on their code), pusb (written by the PUSB authors) and sarpu (written by the SAR-EM authors).
These are the official implementations of the respective methods. PyTorch has been used as deep learning framework to convert the existing methods into deep neural networks.

The folder data contains the different UCI datasets. The image datasets are downloaded.

One can easily install the environment using conda and copying the environment_conda.yml

Moreover, we provide a script experiment.py to reproduce our results. 
The arguments for this script are:
    -nsym : the number of simulations
    -clf : the classifier used  -> 'nn' runs the experiments for image datasets
                                -> 'lr' runs the experiments for UCI datasets
    -strat : label strategy (i.e., S1, S2, S3 or S4) -> you can only use S1 and S4 for the image datasets.
    -ds : the name of the dataset to run the experiments on.
    -device : the GPU device -> if you want to run the image dataset experiment on your Nvidia GPU, you can assign a GPU (i.e., an integer denoting the index of your selected GPU). This parameter is disregarded when running the UCI datasets.

Alternatively, one can run experiment_multi.py which uses multi-processing to run multiple UCI dataset experiments at the same time, but limited to one core per process.

Q4 can be answered by running threshold_experiment.py

Appendix E is reproduced by running test_labeled_unlabeled.py

One can also run our threshold optimizer algorithm, which is included in the threshold_optimizer.py file.
