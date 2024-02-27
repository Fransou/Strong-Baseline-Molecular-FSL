<!--
<script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "Dataset",
    "name": "FS-Mol",
    "description": "A Few-Shot Learning Dataset of Molecules",
    "url": "https://github.com/microsoft/FS-Mol/tree/main/datasets",
    "license": "https://creativecommons.org/licenses/by-sa/3.0/",
    "isAccessibleForFree" : true,
  }
</script>
-->

# Codebase for the paper "Is meta-learning necessary for few-shot molecular learning?"

## :clipboard: Installation

To use this repository, we recommend using a conda environment.
A requirements.txt file is provided to install the required packages,
and the following commands can be used to install the packages.
Pytorch, Torch-geometric and Torch-scatter are also required
(which can be installed using the conda package manager as below).
```bash
pip install -r requirements.txt 
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg -c pyg
conda install pytorch-scatter -c pyg
```
## Data

The FS-mol dataset can be downloaded straight from the FS-mol repository.
To use MHNFS, please refer to the original github of the method to preprocess the data.

*A google drive link will be provided to access data once un-anonymized*

Data for the out-of-distribution experiments can be retrieved using the get_data_TDC notebook, to load the tasks imported from the Therapeutic Data Common Plateform.
To use DATA from the LIT-PCBA paper, select the corresponding dataset from the authors platform (http://drugdesign.unistra.fr/LIT-PCBA) and use the "lit_pcba_to_csv.py" script to convert the data to the format used in the out-of-distribution experiments.

*A google drive link will be provided to access data once un-anonymized*

## Training and evaluation of models on FS-mol 

### Quadratic probe

To train the model, use the following command:

```bash
python fs_mol/qprobe_train.py
```
And to run the evaluation, use the following command:

```bash
python fs_mol/qprobe_eval.py --fold {VALIDATION/TEST}
```

The configuration files of the models are found in the fs_mol/configs/simplebsl directory

### Linear probe

To train or evaluate the model, use the following command:

```bash
python fs_mol/qprobe_train.py
python fs_mol/linear_probe_eval.py --fold {VALIDATION/TEST}
```
The configuration files of the models are found in the fs_mol/configs/LP directory

### CLAMP

To adapt the CLAMP model to the FS-Mol VALIDATION/TEST set, use the following command:

```bash
python fs_mol/clamp_eval.py --fold {VALIDATION/TEST}
```

The pretraining of the CLAMP model was done using the original CLAMP codebase, achieving a 0-shot Delta-AUCPR of 0.20.
*The weights of the models are too large to be uploaded on github*

### MHNFS

To train the model, used the original codebase of MHNFS.
To run the evaluation, use the following command:

```bash
python fs_mol/mhnfs_eval.py
```

The model was pretrained with the authors' codebase, the backbone achieving a Delta-AUCPR of 0.25 on the FS-Mol dataset.

# Out-of-distribution experiments

The directory containing the codebase for the out-of-distribution experiments is found in the TDC_tasks directory.
The directory evaluator contains the code enabling the use of the various models on the TDC tasks (note that MAML and GNN-MT are available although not discussed in the original paper, as they take a long time to run on these tasks)

To run the experiments on the DTI tasks, use the following command:

```bash
python TDC_tasks/main_DTI.py --models {models to evaluate} --task_names {tasks to evaluate}
```

To run the experiments on the Library Screening tasks, use the following command:

```bash
python TDC_tasks/main_HTS.py --models {models to evaluate} --task_names {tasks to evaluate}
```


## Results of the out-of-distribution experiments

The results of the out-of-distribution experiments are found in the TDC_tasks/results directory.
These can be analyzed using the notebooks found in the TDC_tasks directory.


This repisotory is built on the fork of ADKT-IFT (Chen et al. 2023).
find below the original README for the FS-Mol dataset and codebase.


-----------------------------------------------------------------------------------------


-----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
# FS-Mol: A Few-Shot Learning Dataset of Molecules

This repository contains data and code for FS-Mol: A Few-Shot Learning Dataset of Molecules.

## Installation

1. Clone or download this repository
2. Install dependencies

   ```
   cd FS-Mol

   conda env create -f environment.yml
   conda activate fsmol
   ```

The code for the Molecule Attention Transformer baseline is added as a submodule of this repository. Hence, in order to be able to run MAT, one has to clone our repository via `git clone --recurse-submodules`. Alternatively, one can first clone our repository normally, and then set up submodules via `git submodule update --init`. If the MAT submodule is not set up, all the other parts of our repository should continue to work.

## Data

The dataset is available as a download, [FS-Mol Data](https://figshare.com/ndownloader/files/31345321), split into `train`, `valid` and `test` folders. Additionally, we specify which tasks are to be used with the file `datasets/fsmol-0.1.json`, a default list of tasks for each data fold. We note that the complete dataset contains many more tasks. Should use of all possible training tasks available be desired, the training script argument `--task_list_file datasets/entire_train_set.json` should be used. The task lists will be used to version FS-Mol in future iterations as more data becomes available via ChEMBL.

Tasks are stored as individual compressed [JSONLines](https://jsonlines.org/) files, with each line corresponding to the information to a single datapoint for the task.
Each datapoint is stored as a JSON dictionary, following a fixed structure:
```json
{
    "SMILES": "SMILES_STRING",
    "Property": "ACTIVITY BOOL LABEL",
    "Assay_ID": "CHEMBL ID",
    "RegressionProperty": "ACTIVITY VALUE",
    "LogRegressionProperty": "LOG ACTIVITY VALUE",
    "Relation": "ASSUMED RELATION OF MEASURED VALUE TO TRUE VALUE",
    "AssayType": "TYPE OF ASSAY",
    "fingerprints": [...],
    "descriptors": [...],
    "graph": {
        "adjacency_lists": [
           [... SINGLE BONDS AS PAIRS ...],
           [... DOUBLE BONDS AS PAIRS ...],
           [... TRIPLE BONDS AS PAIRS ...]
        ],
        "node_types": [...ATOM TYPES...],
        "node_features": [...NODE FEATURES...],
    }
}
```

### FSMolDataset
The `fs_mol.data.FSMolDataset` class provides programmatic access in Python to the train/valid/test tasks of the few-shot dataset.
An instance is created from the data directory by `FSMolDataset.from_directory(/path/to/dataset)`.
More details and examples of how to use `FSMolDataset` are available in `fs_mol/notebooks/dataset.ipynb`.

## Evaluating a new Model

We have provided an implementation of the FS-Mol evaluation methodology in `fs_mol.utils.eval_utils.eval_model()`.
This is a framework-agnostic python method, and we demonstrate how to use it for evaluating a new model in detail in `notebooks/evaluation.ipynb`.

Note that our baseline test scripts (`fs_mol/baseline_test.py`, `fs_mol/maml_test.py`, `fs_mol/mat_test`, `fs_mol/multitask_test.py` and `fs_mol/protonet_test.py`) use this method as well and can serve as examples on how to integrate per-task fine-tuning in TensorFlow (`maml_test.py`), fine-tuning in PyTorch (`mat_test.py`) and single-task training for scikit-learn models (`baseline_test.py`).
These scripts also support the `--task_list_file` parameter to choose different sets of test tasks, as required.

## Baseline Model Implementations

We provide implementations for three key few-shot learning methods: Multitask learning, Model-Agnostic Meta-Learning, and Prototypical Networks, as well as evaluation on the Single-Task baselines and the Molecule Attention Transformer (MAT) [paper](https://arxiv.org/abs/2002.08264v1), [code](https://github.com/lucidrains/molecule-attention-transformer). 

All results and associated plots are found in the baselines/ directory. 

These baseline methods can be run on the FS-Mol dataset as follows:

### kNNs and Random Forests -- Single Task Baselines

Our kNN and RF baselines are obtained by permitting grid-search over a industry-standard parameter set, detailed in the script `baseline_test.py`.

The baseline single-task evaluation can be run as follows, with a choice of kNN or randomForest model:

```bash
python fs_mol/baseline_test.py /path/to/data --model {kNN, randomForest}
```

### Molecule Attention Transformer

The Molecule Attention Transformer (MAT) [paper](https://arxiv.org/abs/2002.08264v1), [code](https://github.com/lucidrains/molecule-attention-transformer). 

The Molecule Attention Transformer can be evaluated as:

```bash
python fs_mol/mat_test.py /path/to/pretrained-mat /path/to/data
```

### GNN-MAML pre-training and evaluation

The GNN-MAML model consists of a GNN operating on the molecular graph representations of the dataset. The model consists of a $8$-layer GNN with node-embedding dimension $128$. The GNN uses "Edge-MLP" message passing. The model was trained with a support set size of $16$ according to the MAML procedure [Finn 2017](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf). The hyperparameters used in the model checkpoint are default settings of `maml_train.py`.

The current defaults were used to train the final versions of GNN-MAML available here. 

```bash
python fs_mol/maml_train.py /path/to/data 
```

Evaluation is run as: 

```bash
python fs_mol/maml_test.py /path/to/data --trained_model /path/to/gnn-maml-checkpoint
```

### GNN-MT pre-training and evaluation

The GNN-MT model consists of a GNN operating on the molecular graph representations of the dataset. The model consists of a $10$-layer GNN with node-embedding dimension $128$. The model uses principal neighbourhood aggregation (PNA) message passing. The hyperparameters used in the model checkpoint are default settings of `multitask_train.py`. This method has similarities to the approach taken for the task-only training contained within [Hu 2019](https://arxiv.org/abs/1905.12265v1)

```bash
python fs_mol/multitask_train.py /path/to/data 
```

Evaluation is run as: 

```bash
python fs_mol/multitask_test.py /path/to/gnn-mt-checkpoint /path/to/data
```

### Prototypical Networks (PN) pre-training and evaluation

The prototypical networks method [Snell 2017](https://proceedings.neurips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf) extracts representations of support set datapoints and uses these to classify positive and negative examples. We here used the Mahalonobis distance as a metric for query point distance to class prototypes. 

```bash
python fs_mol/protonet_train.py /path/to/data 
```

Evaluation is run as: 

```bash
python fs_mol/protonet_test.py /path/to/pn-checkpoint /path/to/data
```

## Available Model Checkpoints

We provide pre-trained models for `GNN-MAML`, `GNN-MT` and `PN`, these are downloadable from the links to [figshare](https://figshare.com/projects/FS-Mol_Dataset_and_Models/125797).

| Model Name | Description                                                              | Checkpoint File                                                                       |
| ---------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| GNN-MAML   | Support set size 16. 8-layer GNN. Edge MLP message passing.              | [MAML-Support16_best_validation.pkl](https://figshare.com/ndownloader/files/31346701) |
| GNN-MT     | 10-layer GNN. PNA message passing                                        | [multitask_best_model.pt](https://figshare.com/ndownloader/files/31338334)              |
| PN         | 10-layer GGN, PNA message passing. ECFP+GNN, Mahalonobis distance metric | [PN-Support64_best_validation.pt](https://figshare.com/ndownloader/files/31307479)    |


## Specifying, Training and Evaluating New Model Implementations

Flexible definition of few-shot models and single task models is defined as demonstrated in the range of train and test scripts in `fs_mol`. 

We give a detailed example of how to use the abstract class `AbstractTorchFSMolModel` in `notebooks/integrating_torch_models.ipynb` to integrate a new general PyTorch model, and note that the evaluation procedure described below is demonstrated on `sklearn` models in `fs_mol/baseline_test.py` and on a Tensorflow-based GNN model in `fs_mol/maml_test.py`.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
