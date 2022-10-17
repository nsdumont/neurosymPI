# A model of path integration that connects neural and symbolic representation

Path integration, the ability to maintain an estimate of one's location by continuously integrating self-motion cues, is a vital component of the brain's navigation system. We present a spiking neural network model of path integration derived from a starting assumption that the brain represents continuous variables, such as spatial coordinates, using Spatial Semantic Pointers (SSPs). SSPs are a representation for encoding continuous variables as high-dimensional vectors, and can also be used to create structured, hierarchical representations for neural cognitive modelling. Path integration can be performed by a recurrently-connected neural network using SSP representations. Unlike past work, we show that our model can be used to continuously update variables of any dimensionality. We demonstrate that symbol-like object representations can be bound to continuous SSP representations. Specifically, we incorporate a simple model of working memory to remember environment maps with such symbol-like representations situated in 2D space.

### Requriements
This package requries numpy, nengo, matplotlib, scipy, pytry. All can be installed with pip.

### Install
Clone the repo,
```console
git clone git@github.com:nsdumont/neurosymPI.git
```
Then install on your machine,
```console
cd neurosymPI
python3 setup.py install
```

### neurosymPI
The package neurosymPI includes the code used for the experiments in the paper "A model of path integration that connects neural and symbolic representation". It consists of two modules: 
* sspspace: This contains SSPSpace, RandomSSPSpace, and HexagonalSSPSpace. These classes define the SSP vector space. An SSP $S(x)$ is a $d$-dim real-valued vector that represents a $n$-dim varaible x, where $n\ll d$. An SSP vector space is $\mathcal{S} = \\{ S(x)=\mathcal{F}^{-1}\\{e^{iAx/l}\\} \, | \, x \in \mathcal{X} \\}$ where $\mathcal{X}$ is a subspace of $\mathcal{R}^n$, and  $A \in \mathcal{R}^{d \times n}$ is a fixed phase matrix, and $l \in \mathcal{R}^n$ are the length scale parameters. SSPSpace creates a space using a user defined phase matrix $A$, RandomSSPSpace creates a space using a randomly constructed $A$, and HexagonalSSPSpace uses a speacial construction of $A$ for modelling grid cells
* networks: This contains PathIntegration, SSPNetwork, and InputGatedMemory. These are all nengo networks which can be embedded in other networks. PathIntegration does PI with SSPs. SSPNetwork represents SSPs with grid cells. InputGatedMemory is used for the cogntive mapping experiments.

### Experiments 
The tests folder contains simple scripts that use code from neurosymPI.sspspace & neurosymPI.networks. The experiments folder contain several scripts:
* path_generation: This just contains helper functions for generating random paths to test the PI model on
* spiral_path: This creates a neurosymPI.sspspace.HexagonalSSPSpace space for SSPs and uses neurosymPI.networks.PathIntegration to perform PI on a wingly sprial path. Spike train data from the model is saved.
* path_integration_trials: This tests the neurosymPI.networks.PathIntegration model on 10 different random paths and saves the results
* working_memory_map: The cognitive mapping experiment. The PI model is used on a simple elliptical path. Three objects are located along the path. When an object is in view, the output of the PI model is bound to the semantic pointer representing objects idenitty -- creating a joint representation of object type and location. The InputGatedMemory stores these. Also in this script is code for queriing the memory network over time for object locations & the vector between current position and object locations. 
