from kfp.dsl import component, Input, Output, Artifact, Dataset


@component(
    base_image="us-west1-docker.pkg.dev/project-demo-498806/docker/brand-similarity:latest",
    install_kfp_package=False,
)
def calculate_similarity(
    content_weights: str,
    interaction_weights: str,
    global_weights: str,
    content_embeddings: Input[Artifact],
    interaction_embeddings: Input[Artifact],
    brand_data: Input[Dataset],
    output: Output[Dataset],
    reference_dt: str,
    top_n: int = 100,
):
    """Combine content and interaction scores and emit top-N similar brands."""

    import io
    import gcsfs
    import logging
    import numpy as np
    import pandas as pd
    import json
    from itertools import chain
    from sklearn.metrics.pairwise import cosine_similarity
    from utils import preprocess_matrix, map_brand_index

    ####### utils
    def calculate_content_score(
        content_embeddings_path: str,
        content_weights: str,
    ):
        """Load content embeddings and apply configured content weights."""

        def calculate_score(data, weights):
            """Apply content feature weights to loaded matrices."""

            # Apply feature weights.
            scores = {}
            for feature_name in weights.keys():
                if feature_name not in data:
                    logging.warning(
                        f"[WARN] Skipped missing content feature weight: {feature_name}"
                    )
                    continue

                matrix = data[feature_name]

                weight = weights.get(feature_name, 0.1)
                weighted_matrix = weight * matrix

                logging.info(
                    f"[INFO] Standardized and applied weights({feature_name})."
                )
                scores.update({feature_name: weighted_matrix})

            # Compute the aggregate content score.
            score = np.sum(list(scores.values()), axis=0)
            scores.update({"score": score})

            return scores

        # read data
        with fs.open(f"{content_embeddings_path}.npz", "rb") as file:
            data = np.load(io.BytesIO(file.read()), allow_pickle=True)

        weights = json.loads(content_weights)
        scores = calculate_score(data, weights)

        result = {"brand_index": data["brand_index"], "scores": scores}

        return result

    def calculate_interaction_score(
        interaction_embeddings_path: str,
        interaction_weights: str,
    ):
        """Load interaction embeddings and apply configured interaction weights."""

        def calculate_score(data, weights):
            """Map weighted interaction matrices into one shared brand index."""

            all_brands = list(set(chain.from_iterable(data["brand_index"])))

            brand_indices = np.array(sorted(all_brands))
            brand_to_idx = {b: i for i, b in enumerate(brand_indices)}
            N = len(brand_indices)

            feature_list = list(weights.keys())

            # Compute feature-level similarity scores.
            scores = {}
            total_weight = sum(weights.values()) or 1.0

            for feature_name, brand_index, matrix in zip(
                feature_list,
                data["brand_index"],
                data["matrix"],
            ):

                weight = weights.get(feature_name, 1.0)
                weighted_norm = weight / total_weight
                weighted_matrix = matrix * weighted_norm

                # Map local brand indices into the global brand index.
                idx_map = [brand_to_idx[brand_name] for brand_name in brand_index]

                new_matrix = np.zeros((N, N), dtype=np.float32)
                ix_grid = np.ix_(idx_map, idx_map)
                new_matrix[ix_grid] = weighted_matrix

                scores.update({feature_name: new_matrix})

                logging.info(f"[INFO] Calculated score({feature_name}).")

            # Compute the aggregate interaction score.
            score = np.sum(list(scores.values()), axis=0)
            scores.update({"score": score})

            logging.info(f"[INFO] Calculated interaction score.")

            return scores, brand_indices

        # read data
        with fs.open(f"{interaction_embeddings_path}.npz", "rb") as file:
            data = np.load(io.BytesIO(file.read()), allow_pickle=True)

        weights = json.loads(interaction_weights)
        scores, brand_indices = calculate_score(data, weights)

        result = {"brand_index": brand_indices, "scores": scores}

        logging.info(f"[INFO] Saved similarity embeddings.")

        return result

    def calculate_total_score(
        brand_data_path,
        top_n,
        content_data,
        interaction_data,
        global_weights,
        sim_type,
    ):
        """Combine content and interaction scores into a top-N result table."""

        if sim_type != "hybrid":
            global_weights = json.dumps(
                {"interaction_weight": 0.0, "content_weight": 1.0}
            )

        weights = json.loads(global_weights)
        content_weight = weights.get("content_weight")
        interaction_weight = weights.get("interaction_weight")

        interaction_brands = interaction_data["brand_index"].tolist()
        content_brands = content_data["brand_index"].tolist()

        brand_df = pd.read_parquet(brand_data_path)

        # Map all eligible brands into a shared index.
        base_brands = set(interaction_brands) & set(content_brands)
        new_brands = set(brand_df.loc[brand_df.new_brand == "Y", "brand_name"])
        all_brands = base_brands | new_brands

        all_brands = sorted(all_brands)
        brand_to_idx = {brand_name: i for i, brand_name in enumerate(all_brands)}

        content_scores = map_brand_index(data=content_data, brand_to_idx=brand_to_idx)
        interaction_scores = map_brand_index(
            data=interaction_data,
            brand_to_idx=brand_to_idx,
        )

        # Preprocess all-zero rows before cosine similarity.
        content_scores_processed = preprocess_matrix(content_scores)
        interaction_scores_processed = preprocess_matrix(interaction_scores)

        weighted_content = cosine_similarity(content_scores_processed) * content_weight
        weighted_interaction = (
            cosine_similarity(interaction_scores_processed) * interaction_weight
        )
        total_scores = weighted_content + weighted_interaction
        np.fill_diagonal(total_scores, 1.0)

        # Log score ranges for pipeline observability.
        logging.info(
            f"Content Max: {weighted_content.max():.4f} (Target: {content_weight})"
        )
        logging.info(
            f"Interaction Max: {weighted_interaction.max():.4f} (Target: {interaction_weight})"
        )
        logging.info(f"Total Max: {total_scores.max():.4f} (Target: 1.0)")

        # Build the top-N similarity table for each brand.
        similarity_table = []
        brand_indices = np.array(list(all_brands))

        for i, brand_name in enumerate(brand_indices):

            total_score = total_scores[i]
            content_score = weighted_content[i]
            interaction_score = weighted_interaction[i]

            top_n_idx = np.argsort(total_score)[::-1][1 : top_n + 1]

            # Store total and feature-family scores for each top-N brand.
            similar_brands = [
                {
                    "rank": rank + 1,
                    "brand_name": brand_indices[j],
                    "total_score": float(total_score[j]),
                    "content_score": float(content_score[j]),
                    "interaction_score": float(interaction_score[j]),
                }
                for rank, j in enumerate(top_n_idx)
            ]

            # Append the final row for this brand.
            similarity_table.append(
                {
                    "brand_name": brand_name,
                    "similar_brands": brand_indices[top_n_idx],
                    "similarity": similar_brands,
                }
            )

        sim_df = pd.DataFrame(similarity_table)
        sim_df["similarity_type"] = sim_type.upper()
        return sim_df

    ######## main

    fs = gcsfs.GCSFileSystem()

    # Calculate similarity scores.
    content_data = calculate_content_score(
        content_embeddings_path=content_embeddings.uri,
        content_weights=content_weights,
    )

    interaction_data = calculate_interaction_score(
        interaction_embeddings_path=interaction_embeddings.uri,
        interaction_weights=interaction_weights,
    )

    hybrid_score = calculate_total_score(
        content_data=content_data,
        interaction_data=interaction_data,
        global_weights=global_weights,
        brand_data_path=brand_data.path,
        top_n=top_n,
        sim_type="hybrid",
    )

    content_score = calculate_total_score(
        content_data=content_data,
        interaction_data=interaction_data,
        global_weights=global_weights,
        brand_data_path=brand_data.path,
        top_n=top_n,
        sim_type="content",
    )

    df = pd.concat([hybrid_score, content_score])

    brand_df = pd.read_parquet(brand_data.path)
    cols = brand_df.columns[~brand_df.columns.isin(["brand_name", "brand_name_ko"])]
    brand_df["features"] = brand_df[cols].to_dict(orient="records")

    subset = brand_df[["brand_name", "brand_name_ko", "features"]].copy()
    result = subset.merge(df, on="brand_name")

    # json serialization
    class NpEncoder(json.JSONEncoder):
        """JSON encoder for NumPy values in similarity result payloads."""

        def default(self, obj):
            """Convert NumPy arrays and scalars into JSON-serializable values."""
            # Handle numpy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle numpy scalars (like np.int64, np.float32)
            if isinstance(obj, np.generic):
                return obj.item()
            return super(NpEncoder, self).default(obj)

    cols = ["similar_brands", "similarity", "features"]
    result[cols] = result[cols].map(
        lambda x: json.dumps(x, cls=NpEncoder, ensure_ascii=False)
    )

    result["dt"] = reference_dt
    result.columns = result.columns.str.upper()
    result.to_parquet(output.path)
