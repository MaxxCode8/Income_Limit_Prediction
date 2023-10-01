import pandas as pd 
import numpy as np
import joblib 

# loading preprocessing pickels 
target_enc = joblib.load('Model/target_encoder.joblib') 
oe = joblib.load('Model/ordinal_encoder.joblib')
scaler = joblib.load('Model/scaler.joblib')

def ordinal_encoder(split):
    req_list = ["gender","tax_status"]
    split[req_list] = oe.fit_transform(split[req_list])
   
    return split[req_list]

def target_encoder(split):
    req_list = ['education','industry_code_main', 'household_stat']
    print(split[req_list])
    target_enc_df = target_enc.transform(split[req_list])
    
    return target_enc_df

def scale_features(x):
    x_scaled = scaler.fit_transform(x)
    
    return x_scaled