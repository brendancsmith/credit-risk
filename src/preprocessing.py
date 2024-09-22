from pathlib import Path
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder

import pandas as pd

def load_data():
    # Load the dataset
    dataset = Path(__file__).parent / '../data/raw/accepted_2007_to_2018Q4.csv'
    df = pd.read_csv(dataset, low_memory=False)
    
    return df

def drop_sparse_cols(df):
    # Remove columns with more than 50% missing values
    threshold = len(df) * 0.5
    df = df.dropna(thresh=threshold, axis=1)

    return df

def drop_cols(df):
    # Drop columns that are irrelevant or data leakage
    irrelevant_cols = [
        'emp_title', 
        'id', 
        'initial_list_status'
        'member_id', 
        'policy_code', 
        'title', 
        'url',
    ]

    leakage_cols = [
        'collection_recovery_fee', 
        'debt_settlement_flag', 
        'disbursement_method', 
        'funded_amount_inv',
        'funded_amount', 
        'hardship_flag', 
        'issue_d',
        'last_credit_pull_d', 
        'last_fico_range_high'
        'last_fico_range_low', 
        'last_pymnt_amnt',
        'last_pymnt_d', 
        'next_pymnt_d',
        'out_prncp_inv', 
        'out_prncp',
        'pymnt_plan', 
        'recoveries', 
        'total_pymnt_inv', 
        'total_pymnt', 
        'total_rec_int',
        'total_rec_late_fee', 
        'total_rec_prncp', 
    ]

    cols_to_drop = irrelevant_cols + leakage_cols

    df = df.drop(columns=cols_to_drop, errors='ignore')

    return df

def impute_missing_values(df):
    # Fill numerical columns with median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    return df
