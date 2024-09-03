import subprocess
import sys
import importlib
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor as RFR
import pickle
import warnings

# define required libraries
required_libraries = [
    'pandas',
    'numpy',
    'sklearn',
    'pickle',
    'subprocess',
    'sys',
    'importlib',
    'argparse',
]

# ignore unnecessary errors
warnings.filterwarnings('ignore')

# check & install libraries
for lib in required_libraries:
    try:
        importlib.import_module(lib)
    except ImportError:
        print(f"{lib} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

# parsing commander
parser = argparse.ArgumentParser(description=
                                 'AgeMiner: A simple chronological age predictor based on bone mineral density data and basic metadata.'
                                 '\nREADME for this tool: https://github.com/Rarapie/AgeMiner')
parser.add_argument('--input', type=str, required=True, help='Path to the input data.')
parser.add_argument('--model', type=str, required=True, help='Select your targeted model.')
parser.add_argument('--output', type=str, required=True, help='Path to the output.')
args = parser.parse_args()

# input data
input = pd.read_table(args.input, header=0, sep=';')

# load pipeline
with open('models/'+args.model+'.pkl', 'rb') as f:
   pipeline = pickle.load(f)

# predict
preds = pipeline.predict(input)
df = pd.DataFrame({
    'ID': np.arange(1, len(preds) + 1),
    'Prediction': preds
})
print(df)
# output
output_name = args.output
df.to_csv(output_name+'.csv', index=False)