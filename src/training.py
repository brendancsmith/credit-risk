from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

def resample(df):
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)[:2]

    return X_resampled, y_resampled
