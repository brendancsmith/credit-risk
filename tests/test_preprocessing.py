import pytest
import pandas as pd
import numpy as np
from regex import R
from src.preprocessing import (
    drop_sparse_cols,
    drop_cols,
    impute_missing_values,
    convert_dates,
    extract_digits,
    scale_features
)

class TestDropSparseCols:
    def test_drop_sparse_cols(self):
        data = {
            'A': [1, 2, np.nan, np.nan],
            'B': [1, np.nan, np.nan, np.nan],
            'C': [1, 2, 3, 4]
        }
        df = pd.DataFrame(data)
        result_df = drop_sparse_cols(df, missing_rate=0.5)
        assert 'B' not in result_df.columns
        assert 'A' in result_df.columns
        assert 'C' in result_df.columns

class TestDropCols:
    def test_drop_cols(self):
        data = {
            'A': [1, 2, 3, 4],
            'B': [1, 2, 3, 4],
            'C': [1, 2, 3, 4]
        }
        df = pd.DataFrame(data)
        result_df = drop_cols(df, cols=['B', 'C'])
        assert 'B' not in result_df.columns
        assert 'C' not in result_df.columns
        assert 'A' in result_df.columns

class TestImputeMissingValues:
    def test_impute_missing_values(self):
        data = {
            'A': [1, 2, np.nan, 4],
            'B': ['a', 'b', np.nan, 'd']
        }
        df = pd.DataFrame(data)
        result_df = impute_missing_values(df)
        assert result_df['A'].isnull().sum() == 0
        assert result_df['B'].isnull().sum() == 0

class TestConvertDates:
    def test_convert_dates(self):
        data = {
            'date': ['Jan-2020', 'Feb-2020', 'Mar-2020']
        }
        df = pd.DataFrame(data)
        result_df = convert_dates(df, cols=['date'])
        assert result_df['date'].dtype == float

class TestExtractDigits:
    def test_extract_digits(self):
        data = {
            'A': ['abc123', 'def456', 'ghi789']
        }
        df = pd.DataFrame(data)
        result_df = extract_digits(df, cols=['A'])
        assert result_df['A'].tolist() == [123.0, 456.0, 789.0]

class TestScaleFeatures:
    def test_scale_features(self):
        size = 100
        data = {
            'A': np.arange(100),
            'B': np.random.randint(100, size=100),
            'loan_status': [0, 1] * (size // 2) + [0, 1][:size % 2]
        }
        df = pd.DataFrame(data)
        result_df = scale_features(df, 'loan_status')
        assert np.isclose(result_df['A'].mean(), 0)
        assert np.isclose(result_df['A'].std(), 1, atol=1/size)
        assert np.isclose(result_df['B'].mean(), 0)
        assert np.isclose(result_df['B'].std(), 1, atol=1/size)
        assert result_df['loan_status'].tolist()[:4] == [0, 1, 0, 1]
