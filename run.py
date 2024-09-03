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
import time

start_time = time.time()

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
print('================Initializing AgeMiner================')
print('Checking dependencies for AgeMiner...')

# parsing commander
parser = argparse.ArgumentParser(description=
                                 'AgeMiner: A simple chronological age predictor based on bone mineral density data and basic metadata.'
                                 '\nREADME for this tool: https://github.com/Rarapie/AgeMiner')
parser.add_argument('--input', type=str, required=True, help='Path to the input data.')
parser.add_argument('--model', type=str, required=True, help='Select your targeted model.')
parser.add_argument('--output', type=str, required=True, help='Path to the output.')
args = parser.parse_args()

print('AgeMiner is ready for prediction...')

# input data
input = pd.read_csv(args.input, header=0, sep=r'[,;]')

# load pipeline
with open('./models/'+args.model+'.pkl', 'rb') as f:
   pipeline = pickle.load(f)

# predict
preds = pipeline.predict(input)
df = pd.DataFrame({
    'ID': np.arange(1, len(preds) + 1),
    'Prediction': preds
})
print('==================Prediction Result==================')
print(df.to_string(index=False))
# output
output_name = args.output
df.to_csv(output_name+'.csv', index=False)

end_time = time.time()
cost_time = end_time - start_time
print(f'Time cost: {cost_time}s')
print('*ID numbers are in the same order as the input samples')