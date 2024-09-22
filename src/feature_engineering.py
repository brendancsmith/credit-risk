import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

def encode_target(df):
    # Filter df for Fully Paid and Charged Off loans
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]

    # Encode target variable
    df['loan_status'] = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1})

    return df

def encode_categorical(df):
    # Label Encoding for ordinal categorical variables
    df['grade'] = df['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})

    # One-Hot Encoding for nominal categorical variables
    df = pd.get_dummies(df, columns=[
        'application_type',
        'home_ownership',
        'purpose',
        'term',
        'verification_status'
    ])

    # Create a mapping for subgrade
    subgrade_list = sorted(df['sub_grade'].unique())
    subgrade_mapping = {subgrade: idx for idx, subgrade in enumerate(subgrade_list)}
    df['sub_grade'] = df['sub_grade'].map(subgrade_mapping)

    return df

def frequency_encoding(df):
    # Calculate default rate per state
    state_default_rate = df.groupby('addr_state')['loan_status'].mean()

    # Map default rate to each record
    df['state_default_rate'] = df['addr_state'].map(state_default_rate)

    # Drop addr_state
    df = df.drop(columns=['addr_state'])
    
    return df

def convert_dates(df):
    # List of date columns
    date_cols = ['earliest_cr_line']

    # Convert date columns to datetime
    for col in date_cols:
        dates = pd.to_datetime(df[col], format='%b-%Y')
        unix_times = dates.astype(int) / 10**9
        df[col] = unix_times

    return df

def scale_features(df):
    # Identify numerical variables
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numerical_cols.remove("loan_status")  # Exclude target variable

    # Standardization
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def drop_high_corr(df, corr_matrix):
    # Find features with correlation > 0.8
    high_corr_var = np.where(corr_matrix > 0.8)
    high_corr_var = [
        (corr_matrix.columns[x], corr_matrix.columns[y])
        for x, y in zip(*high_corr_var)
        if x != y and x < y
    ]

    # Drop one feature from each pair with correlation > 0.8
    for feature1, feature2 in high_corr_var:
        if feature1 in df.columns:
            df = df.drop([feature1], axis=1)

    return df

def new_features(df):
    # Length of Employment
    df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype(float)
    df['emp_length'].fillna(0, inplace=True)

    # First digits of zip code
    df['zip_code'] = df['zip_code'].str.extract('(\d+)').astype(float)

    return df
