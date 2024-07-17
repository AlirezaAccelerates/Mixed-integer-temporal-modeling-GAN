# Improving mixed-integer temporal modeling by generating synthetic data and using a meta-model: A case study of fluid overload prediction in the intensive care unit 

This repo aims to provide insight into the workflow of the ["Improving mixed-integer temporal modeling by generating synthetic data using conditional generative adversarial networks: a case study of fluid overload prediction in the intensive care unit"](https://www.sciencedirect.com/science/article/abs/pii/S0010482523012143) paper, not a nuanced analysis and step-by-step coding and achieving the results summarized in the paper. That said, you can easily use and expand the provided workflow and draw the appropriate performance results based on your application and available data.

# Highlights
• Addresses gap in modeling mixed-integer temporal data for ICU medication.

• Novel application of synthetic data integration to ICU medication data.

• Uses synthetic data to enhance the model's performance of fluid overload prediction.

• Meta-Learner model can improve the ability to predict positive cases.

• The workflow could improve machine learning models in other ICU outcome predictions.

## Overview
![](Overview.png)

### Objective
The challenge of mixed-integer temporal data, which is particularly prominent for medication use in the critically ill, limits the performance of predictive models. The purpose of this evaluation was to pilot test integrating synthetic data within an existing dataset of complex medication data to improve machine learning model prediction of fluid overload.

### Materials and methods
This retrospective cohort study evaluated patients admitted to an ICU ≥ 72 h. Four machine learning algorithms to predict fluid overload after 48–72 h of ICU admission were developed using the original dataset. Then, two distinct synthetic data generation methodologies (synthetic minority over-sampling technique (SMOTE) and conditional tabular generative adversarial network (CTGAN)) were used to create synthetic data. Finally, a stacking ensemble technique designed to train a meta-learner was established. Models underwent training in three scenarios of varying qualities and quantities of datasets.

### Results
Training machine learning algorithms on the combined synthetic and original dataset overall increased the performance of the predictive models compared to training on the original dataset. The highest performing model was the meta-model trained on the combined dataset with 0.83 AUROC while it managed to significantly enhance the sensitivity across different training scenarios.

### Discussion
The integration of synthetically generated data is the first time such methods have been applied to ICU medication data and offers a promising solution to enhance the performance of machine learning models for fluid overload, which may be translated to other ICU outcomes. A meta-learner was able to make a trade-off between different performance metrics and improve the ability to identify the minority class.

If you are interested, please cite:

```
@article{rafiei2024improving,
  title={Improving mixed-integer temporal modeling by generating synthetic data using conditional generative adversarial networks: A case study of fluid overload prediction in the intensive care unit},
  author={Rafiei, Alireza and Rad, Milad Ghiasi and Sikora, Andrea and Kamaleswaran, Rishikesan},
  journal={Computers in Biology and Medicine},
  volume={168},
  pages={107749},
  year={2024},
  publisher={Elsevier}
}
```
