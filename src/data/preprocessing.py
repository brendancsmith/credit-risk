from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder

def encode_categorical(df):
    # Identify categorical variables
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    categorical_cols.remove("Target")  # Exclude target variable

    # Encode target variable
    df["Target"] = df["Target"].map({1: 0, 2: 1})  # 1: Good Credit (0), 2: Bad Credit (1)

    # Label Encoding for categorical variables
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    return df

def scale_features(df):
    # Identify numerical variables
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numerical_cols.remove("Target")  # Exclude target variable

    # Standardization
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
