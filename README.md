# AgeMiner: A simple chronological age predictor based on bone mineral density data and basic metadata
## 1. Introduction
### 1.1 Overview
In the previous study (doi.org/xxx), we built a series of machine learning models, including Linear Regression, SVR, RFR, XGBoost, and LightGBM, based on 5,134 DXA bone density data samples from the distal 1/3 of the ulna and radius, along with metadata such as gender and BMI. Among the optimal models derived from 30 fine-tuned feature combinations, we selected and packaged the six best models into **AgeMiner**. **AgeMiner** is a command-line prediction program developed using the *pickle* and *argparse* modules, capable of predicting the chronological age of unknown samples.
### 1.2 Requirements
Followings are the system requirements and dependencies needed to run the **AgeMiner**.
|Software |Version |
|-|-|
|Opearting system |Windows, macOS, or Linux |
|Python |3.9 or higher |
|Dependencies | |
|numpy |1.23.5 or higher |
|importlib-metadata |7.1.0 or higher |
|importlib-resources |6.1.0 or higher |
|pandas |2.0.3 or higher |
|scikit-learn |**1.5.1 (mandatory)**|

## 2. Installation
