import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def perform_feature_engineering(data):
    """
    Perform feature engineering on the input data.

    Args:
    - data: Input DataFrame containing the raw network traffic data.

    Returns:
    - feature_engineered_data: DataFrame containing the feature-engineered data.
    """
    # Convert categorical variables to numerical using one-hot encoding
    categorical_cols = ["protocol_type", "service", "flag"]
    data = pd.get_dummies(data, columns=categorical_cols)

    # Normalize numerical features using Min-Max scaling
    numerical_cols = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent",
                      "hot", "num_failed_logins", "num_compromised", "num_root",
                      "num_file_creations", "num_shells", "num_access_files",
                      "num_outbound_cmds", "count", "srv_count", "serror_rate",
                      "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                      "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
                      "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                      "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                      "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                      "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                      "dst_host_srv_rerror_rate"]
    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Create interaction features (example: multiply src_bytes and dst_bytes)
    data["src_dst_bytes_product"] = data["src_bytes"] * data["dst_bytes"]

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=10)  # Adjust the number of components as needed
    pca_features = pca.fit_transform(data[numerical_cols])
    pca_df = pd.DataFrame(pca_features, columns=[f"pca_{i}" for i in range(1, 11)])
    data = pd.concat([data, pca_df], axis=1)

    # Binning numerical features (example: duration)
    data["duration_bin"] = pd.qcut(data["duration"], q=4, labels=False, duplicates='drop')

    return data