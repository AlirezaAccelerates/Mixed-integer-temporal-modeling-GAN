# Fluid overload predicting using a meta-model with the integration of synthetic data 

This repo aims at providing insight into the workflow of the ["Improving mixed-integer temporal modeling by generating synthetic data using conditional generative adversarial networks: a case study of fluid overload prediction in the intensive care unit"]() paper, not a neuanced analysis and step by step coding and acheivng the results summarized in the paper. That's said, you can easily use and expand the provided workflow and draw the appropriated performance results based on your application and avaiable data.

## Overview
![](Overview.png)

# Objective
The challenge of mixed-integer temporal data, which is particularly prominent for medication use in the critically ill, limits the performance of predictive models. The purpose of this evaluation was to pilot test integrating synthetic data within an existing dataset of complex medication data to improve machine learning model prediction of fluid overload. 
# Materials and Methods
This retrospective cohort study evaluated patients admitted to an ICU ≥ 72 hours. Four machine learning algorithms to predict fluid overload after 48-72 hours of ICU admission were developed using the original dataset. Then, two distinct synthetic data generation methodologies (synthetic minority over-sampling technique (SMOTE) and conditional tabular generative adversarial network (CTGAN)) were used to create synthetic data. Finally, a stacking ensemble technique designed to train a meta-learner was established. Models underwent training in three scenarios of varying qualities and quantities of datasets. 
# Discussion 
The integration of synthetically generated data is the first time such methods have been applied to ICU medication data and offers a promising solution to enhance the performance of machine learning models for fluid overload, which may be translated to other ICU outcomes. A meta-learner was able to make a trade-off between different performance metrics and improve the ability to identify the minority class. 

If you are interested, please cite:

```
@ARTICLE{9,
  author={},
  journal={}, 
  title={}, 
  year={},
  volume={},
  number={},
  pages={},
  doi={}}
```