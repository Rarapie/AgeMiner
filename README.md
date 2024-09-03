# AgeMiner: A simple chronological age predictor based on bone mineral density data and basic metadata
## 1. Introduction
### 1.1 Overview
In the previous study (doi.org/xxx), we built a series of machine learning models, including Linear Regression, SVR, RFR, XGBoost, and LightGBM, based on 5,134 DXA bone density data samples from the distal 1/3 of the ulna and radius, along with metadata such as gender and BMI. Among the optimal models derived from 30 fine-tuned feature combinations, we selected and packaged the six best models into **AgeMiner**. **AgeMiner** is a command-line prediction program developed using the *pickle* and *argparse* modules, capable of predicting the chronological age of unknown samples.
### 1.2 Requirements
Following are the system requirements and dependencies needed to run the **AgeMiner**.
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
Please clone the repository, and run with cmd.
```python
git clone https://github.com/Rarapie/AgeMiner.git
```
After cloning the repository, please unzip the **models.zip** in the 'models' folder. For example, the path of 'model1' should be './models/model1.pkl'
 ## 3. Usage
Following are the basic commands in **AgeMiner**.
|Software |Version |
|-|-|
|--help/-h |View descriptions of all commands |
|--input |Designate the path to the input file |
|--model |Select 'model1' ~ 'model6' (please refer to section 4) |
|--output |Designate the path to the output file |
### 3.1 Example
```python
python run.py --input [INPUT] --model model[1-6] --output [OUTPUT]
```
`Tips: .csv format input file with 'sep = ;' or 'sep = ,' is recommended.`  
`For details about the format of the input file, refer to 'demo/example.csv'`
## 4. Model selection
Please select the model referring to the following flowchart according to your data situation.
![image](https://github.com/Rarapie/AgeMiner/blob/main/flowchart.png)

