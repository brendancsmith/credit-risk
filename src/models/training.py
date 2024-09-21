from sklearn.model_selection import train_test_split

def encode_y(df, target_column):
    # y is provided in the set [1,2] and we want to transform it to [0,1]
    df[target_column] = df[target_column].apply(lambda x: 0 if x == 1 else 1)
    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
