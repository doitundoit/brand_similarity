from kfp.dsl import component, Output, Artifact, Input, Dataset


@component(
    base_image="us-west1-docker.pkg.dev/project-demo-498806/docker/brand-similarity:latest",
    install_kfp_package=False,
)
def process_interaction_embeddings(
    reference_dt: str,
    brand_data: Input[Dataset],
    output: Output[Artifact],
    period: int = 90,
):
    """Build interaction-based similarity matrices from demo behavior events."""

    from typing import Dict
    from google.cloud import bigquery
    import pandas as pd
    import numpy as np
    import asyncio
    import logging
    from time import perf_counter

    from scipy.sparse import coo_matrix, csr_matrix
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity

    # Feature Information
    FEATURE_INFO = [
        {
            "feature_name": "search_to_click",
            "event_name": "click_search_result",
            "row_id": "session_id",
            "col_id": "brand_name",
        },
        {
            "feature_name": "co_purchase",
            "event_name": "purchase",
            "row_id": "order_id",
            "col_id": "brand_name",
        },
        {
            "feature_name": "co_view",
            "event_name": "view_product",
            "row_id": "session_id",
            "col_id": "brand_name",
        },
        {
            "feature_name": "shared_carts",
            "event_name": "add_to_cart",
            "row_id": "distinct_id",
            "col_id": "brand_name",
        },
        {
            "feature_name": "shared_purchases",
            "event_name": "purchase",
            "row_id": "distinct_id",
            "col_id": "brand_name",
        },
    ]

    # SQL Queries
    SQL_EVENT_PAIRS = """
    WITH USER_BRAND AS (
        SELECT  src_context_key AS row_id
        ,       src_entity_key AS brand_name
        FROM `project-demo-498806.sample.demo_brand_interactions`
        WHERE 1=1
        AND src_event_name = '{event_name}'
        AND src_event_dt >= PARSE_DATE('%Y%m%d', '{reference_dt}') - INTERVAL {period} DAY
        AND src_event_dt < PARSE_DATE('%Y%m%d', '{reference_dt}')
        AND TRIM(src_entity_key) IS NOT NULL
        GROUP BY 1, 2
        HAVING brand_name <> ''
    ),
    TOTAL_USERS AS (
        SELECT COUNT(DISTINCT row_id) AS total_rows FROM USER_BRAND
    )
    SELECT U.row_id, U.brand_name, T.total_rows
    FROM USER_BRAND U
    CROSS JOIN TOTAL_USERS T
    """

    SQL_CO_VIEW_CO_OCCUR = """
    WITH TOTAL_USERS AS (
        SELECT COUNT(DISTINCT src_context_key) AS total_rows
        FROM `project-demo-498806.sample.demo_brand_interactions`
        WHERE src_event_name = 'view_product'
        AND src_event_dt >= PARSE_DATE('%Y%m%d', '{reference_dt}') - INTERVAL {period} DAY
        AND src_event_dt < PARSE_DATE('%Y%m%d', '{reference_dt}')
    )
    SELECT  src_pair_left AS brand_a
    ,       src_pair_right AS brand_b
    ,       src_pair_count AS co_count
    ,       T.total_rows
    FROM `project-demo-498806.sample.demo_brand_pair_counts`
    CROSS JOIN TOTAL_USERS T
    WHERE src_event_name = 'view_product'
    AND src_event_dt >= PARSE_DATE('%Y%m%d', '{reference_dt}') - INTERVAL {period} DAY
    AND src_event_dt < PARSE_DATE('%Y%m%d', '{reference_dt}')
    AND TRIM(src_pair_left) IS NOT NULL
    AND TRIM(src_pair_right) IS NOT NULL
    """

    def get_data(client, feature: Dict, brand_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch behavioral data from BigQuery based on the feature configuration.
        """
        event_name = feature.get("event_name")
        feature_name = feature.get("feature_name")
        row_id = feature.get("row_id")

        if feature_name == "co_view":
            sql = SQL_CO_VIEW_CO_OCCUR.format(
                reference_dt=reference_dt,
                period=period,
            )

        else:
            sql = SQL_EVENT_PAIRS.format(
                event_name=event_name,
                reference_dt=reference_dt,
                period=period,
            )

        df = client.query(sql).to_dataframe()

        if {"brand_a", "brand_b", "co_count"}.issubset(df.columns):
            unique_brands = (
                pd.concat([df["brand_a"], df["brand_b"]]).nunique()
                if not df.empty
                else 0
            )
            logging.info(
                "[INFO] Loaded %s(%s) co-occurrence data: row_id=%s, co_pairs=%s, unique_brands=%s, total_rows=%s.",
                feature_name,
                event_name,
                row_id,
                len(df),
                unique_brands,
                df["total_rows"].iloc[0] if not df.empty else 0,
            )
        else:
            logging.info(
                "[INFO] Loaded %s(%s) data: row_id=%s, pairs=%s, unique_rows=%s, unique_brands=%s, total_rows=%s.",
                feature_name,
                event_name,
                row_id,
                len(df),
                df["row_id"].nunique() if not df.empty else 0,
                df["brand_name"].nunique() if not df.empty else 0,
                df["total_rows"].iloc[0] if not df.empty else 0,
            )

        return df

    def build_cooccurrence_matrix(
        pairs: pd.DataFrame,
        brand_index,
    ):
        """
        Build brand-brand co-occurrence with one vote per row_id/brand pair.
        This is equivalent to the previous BigQuery USER_BRAND self-join.
        """
        brand_to_idx = {brand_name: idx for idx, brand_name in enumerate(brand_index)}
        deduped = pairs.loc[:, ["row_id", "brand_name"]].dropna().drop_duplicates()
        deduped = deduped[deduped["brand_name"].isin(brand_to_idx)]

        if deduped.empty:
            return csr_matrix((len(brand_index), len(brand_index)), dtype=np.float32)

        row_codes = pd.factorize(deduped["row_id"], sort=False)[0]
        col_codes = deduped["brand_name"].map(brand_to_idx).to_numpy()
        data = np.ones(len(deduped), dtype=np.float32)

        user_brand = coo_matrix(
            (data, (row_codes, col_codes)),
            shape=(row_codes.max() + 1, len(brand_index)),
        ).tocsr()

        return (user_brand.T @ user_brand).astype(np.float32).tocsr()

    def build_cooccurrence_matrix_from_counts(
        counts_df: pd.DataFrame,
        brand_index,
    ):
        """Build a sparse brand-brand matrix from pre-aggregated pair counts."""
        brand_to_idx = {brand_name: idx for idx, brand_name in enumerate(brand_index)}
        valid_mask = counts_df["brand_a"].isin(brand_to_idx) & counts_df[
            "brand_b"
        ].isin(brand_to_idx)
        df_valid = counts_df[valid_mask]

        if df_valid.empty:
            return csr_matrix((len(brand_index), len(brand_index)), dtype=np.float32)

        row_indices = df_valid["brand_a"].map(brand_to_idx).to_numpy()
        col_indices = df_valid["brand_b"].map(brand_to_idx).to_numpy()
        data = df_valid["co_count"].to_numpy(dtype=np.float32)

        return coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(brand_index), len(brand_index)),
        ).tocsr()

    def normalize_product_matrix(feature_name, product_raw, product, total_rows):
        """Normalize a co-occurrence matrix according to the feature semantics."""
        if feature_name in ["search_to_click", "co_view"]:
            normalized = product.copy().astype(np.float32).tocsr()
            normalized.data = np.log1p(normalized.data)
            max_val = normalized.data.max() if normalized.nnz else 0.0
            if max_val > 0:
                normalized.data = normalized.data / max_val

        elif feature_name in ["shared_carts", "shared_purchases"]:
            normalized = product_raw.copy().astype(np.float32).tocoo()
            counts = product_raw.diagonal().astype(np.float32)
            union = counts[normalized.row] + counts[normalized.col] - normalized.data
            normalized.data = normalized.data / (union + 1e-12)
            normalized = normalized.tocsr()

        else:
            denominator = max(int(total_rows), 1)
            normalized = product.copy().astype(np.float32).tocoo()
            freq = product.diagonal().astype(np.float32) / denominator
            p_ab_vals = normalized.data / denominator
            pmi_vals = np.log(p_ab_vals + 1e-12) - (
                np.log(freq[normalized.row] + 1e-12)
                + np.log(freq[normalized.col] + 1e-12)
            )
            normalized.data = pmi_vals / (-np.log(p_ab_vals + 1e-12) + 1e-12)
            normalized = normalized.tocsr()

        normalized.setdiag(1.0)
        normalized.eliminate_zeros()
        return normalized.tocsr()

    def compute_matrix(df: pd.DataFrame, info: Dict, brand_df: pd.DataFrame):
        """
        Compute the interaction matrix from user-brand pairs with sparse operations.
        """
        feature_name = info.get("feature_name")
        brand_index = brand_df["brand_name"].unique().tolist()
        brand_index = [b.strip() for b in brand_index]
        total_rows = int(df["total_rows"].iloc[0]) if not df.empty else 0

        if {"brand_a", "brand_b", "co_count"}.issubset(df.columns):
            product_raw = build_cooccurrence_matrix_from_counts(df, brand_index)
        else:
            product_raw = build_cooccurrence_matrix(df, brand_index)

        product = product_raw
        normalized = normalize_product_matrix(
            feature_name=feature_name,
            product_raw=product_raw,
            product=product,
            total_rows=total_rows,
        )

        logging.info(
            "[INFO] Computed the interaction matrix(%s): shape=%s, nnz=%s.",
            feature_name,
            normalized.shape,
            normalized.nnz,
        )

        return (feature_name, brand_index, normalized)

    def find_best_n(X) -> int:
        """
        Find the optimal number of components for SVD using the elbow method.
        """
        n_features = X.shape[1]
        if n_features <= 2 or X.nnz == 0:
            return 1

        n_components = min(n_features - 1, 300)
        model = TruncatedSVD(
            algorithm="randomized",
            n_components=n_components,
            random_state=42,
        )

        model.fit(X)
        cumulative_variances = np.cumsum(model.explained_variance_ratio_)
        if cumulative_variances.size == 0:
            return 1

        cumulative_variances /= cumulative_variances[-1] + 1e-12

        n_points = len(cumulative_variances)
        x = np.arange(n_points)
        y = cumulative_variances

        # Line vector from the first point to the last point.
        first = np.array([0, y[0]])
        last = np.array([n_points - 1, y[-1]])

        line_vec = last - first
        norm = np.linalg.norm(line_vec)
        if norm == 0:
            return 1
        line_vec /= norm

        # Distances between each point and the fitted line.
        all_points = np.vstack((x, y)).T
        vec_from_first = all_points - first
        proj_len = np.dot(vec_from_first, line_vec)
        proj = np.outer(proj_len, line_vec)
        dist = np.linalg.norm(vec_from_first - proj, axis=1)

        best_n = int(np.argmax(dist) + 1)
        return best_n

    def tsvd_matrix(X):
        """
        Apply TruncatedSVD to the feature matrix.
        """
        X = X.astype(np.float32).tocsr()
        if X.nnz == 0:
            return np.zeros((X.shape[0], 1), dtype=np.float32)

        n = find_best_n(X)

        tsvd = TruncatedSVD(
            algorithm="randomized",
            n_components=n,
            random_state=42,
        )
        X_tsvd = tsvd.fit_transform(X).astype(np.float32)

        logging.info(f"[INFO] Applied Truncated SVD to the feature matrix.")

        return X_tsvd

    async def compute_interaction_features(client, brand_df, info):
        """
        Async wrapper to fetch data, compute matrix, and apply TSVD.
        """
        loop = asyncio.get_running_loop()
        feature_name = info.get("feature_name")

        started = perf_counter()
        df = await loop.run_in_executor(None, get_data, client, info, brand_df)
        loaded = perf_counter()

        feature_name, brand_index, X = await loop.run_in_executor(
            None, compute_matrix, df, info, brand_df
        )
        matrix_done = perf_counter()

        X_tsvd = await loop.run_in_executor(None, tsvd_matrix, X)
        svd_done = perf_counter()

        sim_matrix = cosine_similarity(X_tsvd)
        finished = perf_counter()

        logging.info(
            "[INFO] Timings(%s): query=%.2fs matrix=%.2fs svd=%.2fs cosine=%.2fs total=%.2fs.",
            feature_name,
            loaded - started,
            matrix_done - loaded,
            svd_done - matrix_done,
            finished - svd_done,
            finished - started,
        )

        return (feature_name, brand_index, sim_matrix)

    async def main():
        """Run all interaction feature tasks and save the embedding artifact."""
        client = bigquery.Client(project="project-demo-498806")
        brand_df = pd.read_parquet(brand_data.path)
        tasks = [
            compute_interaction_features(client, brand_df, info)
            for info in FEATURE_INFO
        ]

        results = await asyncio.gather(*tasks)
        feature_names, brand_indices, matrices = zip(*results)

        np.savez_compressed(
            output.path,
            feature_name=np.array(feature_names, dtype="object"),
            brand_index=np.array(brand_indices, dtype="object"),
            matrix=np.array(matrices, dtype="object"),
        )
        logging.info("[INFO] Saved interaction embeddings.")

    asyncio.run(main())
