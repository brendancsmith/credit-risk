def correlation_matrix(df):
    # Convert non-numeric columns to numeric or drop them
    df_numeric = df.select_dtypes(include=[float, int])

    # Compute correlation matrix
    corr = df_numeric.corr()

    return corr
