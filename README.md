# ensgendel & incremental evaluation framework
This package contains implementation of **ens**emble of **gen**erative models with **del**etion mechanism (ensgendel)
 and the framework for continual learning evaluation.
 
## Requirements
For running ensgendel with GPU, [cupy](https://cupy.dev/) is required.
Tested on python 3.7.4. (only CPU) and 2.7. (with GPU). Excerpts from `pip freeze` command are below.

Python 3.7.4. setup:\
chainer==7.7.0
h5py==2.9.0
matplotlib==3.0.2
numpy==1.17.2
scikit-build==0.11.1
scikit-learn==0.21.3
scipy==1.3.1

Python 2.7. setup:\
chainer==5.2.0
cupy==5.2.0
h5py==2.7.1
matplotlib==2.1.0
numpy==1.14.0
scikit-learn==0.19.1
scipy==1.0.0

## Run demo
1. From the project root run the demo application *incremental_evaluation_run.py*\
`python incremental_evaluation_run.py exp1 mnist_cn5 12345 --debug True`\
The application creates *results/incremental_evaluation_run/* file where app stores results of *exp1* experiment.
The `--debug True` option runs only ligthweight classifiers, so this run just tests whether everything works.
2. If no errors showed up in from previous task run\
`python incremental_evaluation_run.py exp2 mnist012 234 --scout 1000 -trials 3`\
which creates experiment *exp2* where all predictors are evaluated on *mnist012* scenarios, which operate on pruned
dataset. The results are averaged from three trials.
3. For more options consult the help option\
`python incremental_evaluation_run.py`

If you have CUDA and *cupy* installed, turn on the GPU by 
## Adding new continual learners and scenarios
You can integrate new continual learners and scenarios by implementing the interfaces **Predictor** and **ScenarioSet**,
 respectively. See interface definitions in *incremental_evaluation/interfaces* for details.