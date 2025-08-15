import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(data_path, seed):
    df = pd.read_csv(data_path)
    df = df.drop(columns=['id', 'date', 'zipcode'])
    # Label the price into 4 quartile-based bins and get bin intervals
    df['price_class'], bins = pd.qcut(df['price'], q=4, labels=[0, 1, 2, 3], retbins=True, duplicates='drop')
    df['price_class'] = df['price_class'].astype(int)

    # Print bin ranges
    print("\nPrice class ranges:")
    for i in range(len(bins) - 1):
        print(f"Class {i}: ${bins[i]:,.0f} - ${bins[i+1]:,.0f}")

    
    # Check distribution
    print("Price label distribution:")
    print(df['price_class'].value_counts().sort_index())

    X = df.drop(columns=['price', 'price_class']).values
    y = df['price_class'].values  # Use quartile class labels


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
