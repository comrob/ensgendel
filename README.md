# ensgendel & incremental evaluation framework
This package contains implementation of **ens**emble of **gen**erative models with **del**etion mechanism (ensgendel)
 and the framework for continual learning evaluation.
 
## Requirements
Tested on python 3.7.4. (only CPU) and 2.7. (with GPU).

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
From the project root run the demo application *incremental_evaluation_run.py*\
some command \
The application creates results/incremental_evaluation_run/ file where the app stores all results.

## Adding new continual learners and scenarios
You can integrate new continual learners and scenarios by implementing the interfaces **Predictor** and **ScenarioSet**,
 respectively. See interface definitions in *incremental_evaluation/interfaces* for details.