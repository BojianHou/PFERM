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
conda create -n PFERM python==3.11    # create an environment named "PFERM"
conda activate PFERM                  # activate environment
pip install -r requirements.txt       # install required packages
```

### 🔨 Usage

To run the code please use 
```bash 
python main.py
``` 
with specific arguments as described.

### 🤝 Acknowledgements

This work was supported in part by the NIH grants U01 AG066833, U01 AG068057, P30 AG073105, RF1 AG063481 and U01 CA274576. The ADNI data were obtained from the Alzheimer’s Disease Neuroimaging Initiative database (https://adni.loni.usc.edu), funded by NIH U01 AG024904.

### 📭 Maintainers

[Bojian Hou](http://bojianhou.com) 
- ([bojian.hou@pennmedicine.upenn.edu](mailto:bojian.hou@pennmedicine.upenn.edu))
- ([hobo.hbj@gmail.com](mailto:hobo.hbj@gmail.com))


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
