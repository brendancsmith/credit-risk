import numpy as np


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
