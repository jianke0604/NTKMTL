# [NeurIPS’25] NTKMTL

The official implementation of our NeurIPS 2025 paper:  
- [NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective](https://arxiv.org/abs/2510.18258)  

## Setup Environment
First, use miniconda to create a new environment:
```bash
conda create -n mtl python=3.9.7
conda activate mtl
```
Regarding the torch version in the environment, no stringent constraints are imposed; minor adjustments may be made based on the locally installed CUDA version.

Then, install the repository :
```bash
git clone https://github.com/jianke0604/NTKMTL.git
cd NTKMTL
pip install -r requirements.txt
```


## Download Datasets
The performance is evaluated under 3 scenarios:
 - Image-level Classification. The [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains 40 tasks.
 - Regression. The QM9 dataset contains 11 tasks, which can be downloaded automatically from Pytorch Geometric.
 - Dense Prediction. The [NYU-v2](https://github.com/lorenmt/mtan) dataset contains 3 tasks and the [Cityscapes](https://github.com/lorenmt/mtan) dataset contains 2 tasks.

## Run Experiments
To run the experiments, use the following command:
```bash
cd experiments/EXP_NAME
sh run.sh
```
For example, the `run.sh` script in `experiments/quantum_chemistry` contains the following command:
```bash
mkdir -p ./save
mkdir -p ./trainlogs

export PYTHONPATH=$PYTHONPATH:YOUR_PYTHON_PATH
export CUDA_VISIBLE_DEVICES=0

method=ntkmtl
seed=0
ntk_exp=0.5

python -u trainer.py \
 --method=$method \
 --seed=$seed  \
 --ntk_exp=$ntk_exp \
 --wandb_logger_name "XXX" \
 --wandb_project=XXX \
 --wandb_entity=XXX
```

## Implementation Details

Our approach includes two variants: NTKMTL and NTKMTL-SR.  For NTKMTL, we set the hyperparameter *n* (see the main text for details) to 1 in order to ensure a fair comparison with other gradient-oriented methods.  For NTKMTL-SR, *n* is set to 4 by default.  These two variants differ substantially in their implementation.  By default, the library provides the **NTKMTL** implementation.  To switch to **NTKMTL-SR**, please refer to the commented sections in `weight_methods.py` and the corresponding experiment scripts in `trainer.py`, and modify them accordingly.

In addition, in the main text, we apply a square root to the NTK ratio to balance the convergence rate across tasks.  However, we empirically observed that using a slightly larger exponent (e.g., 0.75) can sometimes yield better performance, as it encourages the algorithm to focus more on harder-to-converge tasks.  Therefore, we introduce a hyperparameter `ntk_exp`, which controls the exponent applied to the NTK ratio.  By default, `ntk_exp = 0.5`, but we recommend exploring slightly larger values in experiments, as they may lead to improved performance in certain cases.


## Acknowledgements
This codebase is built upon [FairGrad](https://github.com/OptMN-Lab/fairgrad) and [MTLlib](https://github.com/jianke0604/MTLlib). We sincerely thank the authors for their efforts and contributions.

## Citation
If you find this repository helpful, please consider citing our paper:
```bibtex
@article{qin2025ntkmtl,
  title={NTKMTL: Mitigating Task Imbalance in Multi-Task Learning from Neural Tangent Kernel Perspective},
  author={Qin, Xiaohan and Wang, Xiaoxing and Liao, Ning and Yan, Junchi},
  journal={arXiv preprint arXiv:2510.18258},
  year={2025}
}
```
