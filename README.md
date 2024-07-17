# PFERM: A Fair Empirical Risk Minimization Approach with Prior Knowledge

This repository holds the official code for the paper 

* "[*PFERM: A Fair Empirical Risk Minimization Approach with Prior Knowledge*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11141835/)", 
 published in [AMIA 2024 Informatics Summit](https://amia.org/education-events/amia-2024-informatics-summit).  



### 🦸‍♀ Abstract
Fairness is crucial in machine learning to prevent bias based on sensitive attributes in classifier predictions. However, the pursuit of strict fairness often sacrifices accuracy, particularly when significant prevalence disparities exist among groups, making classifiers less practical. For example, Alzheimer’s disease (AD) is more prevalent in women than men, making equal treatment inequitable for females. Accounting for prevalence ratios among groups is essential for fair decision-making. In this paper, we introduce prior knowledge for fairness, which incorporates prevalence ratio information into the fairness constraint within the Empirical Risk Minimization (ERM) framework. We develop the Prior-knowledge-guided Fair ERM (PFERM) framework, aiming to minimize expected risk within a specified function class while adhering to a prior-knowledge-guided fairness constraint. This approach strikes a flexible balance between accuracy and fairness. Empirical results confirm its effectiveness in preserving fairness without compromising accuracy.


### 📝 Requirements

All required libraries are included in the conda environment specified by 
[`requirements.txt`](requirements.txt). To install and activate it, follow the instructions below:

```
conda create -n DCSM               # create an environment named "DCSM"
conda activate DCSM                # activate environment
pip install -r requirements.txt       # install required packages
```

### 🔨 Usage

File [`main.py`](main.py) trains and evaluates the DCSM model. 
It accepts following arguments:

```
  --dataset DATASET     dataset in [sim, support, flchain, PBC, FRAMINGHAM]
  --is_normalize IS_NORMALIZE
                        whether to normalize data
  --is_cluster IS_CLUSTER
                        whether to use DCSM to do clustering
  --is_generate_sim IS_GENERATE_SIM
                        whether we generate simulation data
  --is_save_sim IS_SAVE_SIM
                        whether we save simulation data
  --num_inst NUM_INST   specifies the number of instances for simulation data
  --num_feat NUM_FEAT   specifies the number of features for simulation data
  --cuda_device CUDA_DEVICE
                        specifies the index of the cuda device
  --discount DISCOUNT   specifies number of discount parameter
  --weibull_shape WEIBULL_SHAPE
                        specifies the Weibull shape
  --num_cluster NUM_CLUSTER
                        specifies the number of clusters
  --train_DCSM TRAIN_DCSM
                        whether to train DCSM
```

* The DCSM model is implemented in [`models/dcsm_torch.py`](models/dcsm_torch.py) which
includes definitons for the Deep Clustering Survival Machines module.
The main interface is the DeepClusteringSurvivalMachines class which inherits
from torch.nn.Module. 
* [`models/dcsm_api.py`](models/dcsm_api.py) is a wrapper 
around torch implementations and provides a convenient API to train 
Deep Clustering Survival Machines.
* [`utils/model_utils.py`](utils/model_utils.py) provides several functions 
for model training utilities.
* Data are provided in the [`datasets`](datasets) folder, 
which includes four real-world datasets including support, 
flchain, PBC and FRAMINGHAM that are presented in our paper. 
* [`utils/data_utils.py`](utils/data_utils.py) provides the data loader 
to load these datasets mentioned above. 
We also provide the functions to generate synthetic data in this file. 
* In [`utils/losses.py`](utils/losses.py), we define 
various losses for the censored and uncensored
instances of data corresponding to Weibull distribution. 
* [`utils/plottings.py`](utils/plottings.py) provides several functions to 
plot figures such as the Kaplan-Meier curves.
* [`utils/general_utils.py`](utils/general-utils.py) provides several helper functions 
for model training and testing.

### 🤝 Acknowledgements

- The real-world datasets and their utility functions for data preprocessing 
were taken from Nagpal *et al.*'s and 
[auton-survival repository](https://github.com/autonlab/auton-survival) and 
Manduchi *et al.*'s [vadesc repository](https://github.com/i6092467/vadesc).
- The generation process of synthetic data follows Manduchi *et al.*'s 
[vadesc repository](https://github.com/i6092467/vadesc).

### 📭 Maintainers

[Bojian Hou](http://bojianhou.com) 
- ([bojian.hou@pennmedicine.upenn.edu](mailto:bojian.hou@pennmedicine.upenn.edu))
- ([hobo.hbj@gmail.com](mailto:hobo.hbj@gmail.com))


### 📚 References

Below are some important references that inspires our work:
- Chirag Nagpal, Xinyu Li, and Artur Dubrawski, “Deep
survival machines: Fully parametric survival regression
and representation learning for censored data with competing risks,” 
**IEEE Journal of Biomedical and Health
Informatics**, vol. 25, no. 8, pp. 3163–3175, 2021.
- Laura Manduchi, Riˇcards Marcinkeviˇcs, Michela C
Massi, Thomas Weikert, Alexander Sauter, Verena
Gotta, Timothy M ̈uller, Flavio Vasella, Marian C Neidert, 
Marc Pfister, et al., “A deep variational approach to clustering survival data,” in
**Proceedings of the Tenth International Conference on Learning Representations**, 2022.
- Paidamoyo Chapfuwa, Chunyuan Li, Nikhil Mehta,
Lawrence Carin, and Ricardo Henao, “Survival cluster analysis,” 
in **Proceedings of the ACM Conference on Health, Inference, and Learning**, 
2020, pp. 60–68.


### 🙂 Citation

```
@article{hou2024pferm,
  title={PFERM: A Fair Empirical Risk Minimization Approach with Prior Knowledge},
  author={Hou, Bojian and Mondrag{\'o}n, Andr{\'e}s and Tarzanagh, Davoud Ataee and Zhou, Zhuoping and Saykin, Andrew J and Moore, Jason H and Ritchie, Marylyn D and Long, Qi and Shen, Li},
  journal={AMIA Summits on Translational Science Proceedings},
  volume={2024},
  pages={211},
  year={2024},
  publisher={American Medical Informatics Association}
}
```
