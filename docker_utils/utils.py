import numpy as np


def preprocess_matrix(matrix, epsilon=0.01):
    """
    Add epsilon to low-norm rows before cosine similarity.

    Rows and columns for unmapped brands can remain all zeros after
    map_brand_index. Fully zero vectors may trigger divide-by-zero,
    overflow, or NaN/Inf values in downstream similarity calculations.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1)

    # Add epsilon to the diagonal for low-norm rows.
    for i in np.where(norms < epsilon)[0]:
        matrix[i, i] = epsilon

    return matrix


def map_brand_index(data, brand_to_idx):
    """
    Map a local brand-indexed score matrix to the global brand index.
    """
    # Source brand order must match the score matrix order.
    brand_index = data["brand_index"].tolist()

    feature_matrix = data["scores"]
    matrix = feature_matrix["score"]

    N = len(brand_to_idx)

    # Collect matching source and destination indices.
    src_indices = []
    dst_indices = []

    for i, brand in enumerate(brand_index):
        if brand in brand_to_idx:
            src_indices.append(i)
            dst_indices.append(brand_to_idx[brand])

    # Create an empty matrix at the global size.
    mapped = np.zeros((N, N), dtype=np.float32)

    # Copy the local submatrix into the global matrix when matches exist.
    if src_indices:
        mapped[np.ix_(dst_indices, dst_indices)] = matrix[
            np.ix_(src_indices, src_indices)
        ]

    return mapped
