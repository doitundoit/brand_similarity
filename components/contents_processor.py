from kfp.dsl import component, Output, Artifact, Dataset, Input


@component(
    base_image="us-west1-docker.pkg.dev/project-demo-498806/docker/brand-similarity:latest",
    install_kfp_package=False,
)
def process_content_embeddings(brand_data: Input[Dataset], output: Output[Artifact]):
    """Build content-based similarity matrices from brand profile features."""

    import pandas as pd
    import numpy as np
    import logging

    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    from gensim.models.fasttext import FastText

    def js_divergence_matrix(X):
        """
        Compute a Jensen-Shannon divergence matrix and convert it to similarity.
        """
        X = np.clip(X, 1e-12, 1)
        X = X / X.sum(axis=1, keepdims=True)
        M = 0.5 * (X[:, None, :] + X[None, :, :])
        KL1 = np.sum(X[:, None, :] * np.log(X[:, None, :] / M), axis=-1)
        KL2 = np.sum(X[None, :, :] * np.log(X[None, :, :] / M), axis=-1)
        JS = 0.5 * (KL1 + KL2)
        return 1 - JS

    def train_words(series):
        """Train a FastText model from non-empty category term lists."""
        sentences = [
            cats.tolist() if isinstance(cats, np.ndarray) else cats
            for cats in series
            if isinstance(cats, (list, np.ndarray)) and len(cats) > 0
        ]
        model = FastText(
            sentences, sg=1, vector_size=50, window=3, min_count=1, workers=2
        )
        return model

    def avg_word_vecs(words, model):
        """
        Summarize a list of terms as one representative FastText vector.
        """

        # Return a zero vector for missing or empty term lists.
        if not isinstance(words, (list, np.ndarray)) or len(words) == 0:
            return np.zeros(model.vector_size)

        vecs = [model.wv[w] for w in words if w in model.wv]

        if vecs:
            return np.mean(vecs, axis=0)
        else:
            # The list is non-empty, but none of its terms exist in the vocabulary.
            return np.zeros(model.vector_size)

    def words_embedding(series):
        """Convert each term list in a series into an averaged FastText vector."""
        model = train_words(series)
        vecs = np.vstack([avg_word_vecs(cats, model) for cats in series])
        return vecs

    def compute_content_features(df):
        """
        Compute content-based feature matrices.
        """
        # Embed category_group terms.
        cat_group_vecs = words_embedding(df["category_group"])

        # Embed description sentences.
        df.loc[df["description"].isna(), "description"] = ""
        model = SentenceTransformer(
            "intfloat/multilingual-e5-base", similarity_fn_name="cosine"
        )
        desc_vecs = model.encode(df["description"].tolist(), normalize_embeddings=True)

        logging.info("[INFO] Completed embeddings calculation.")

        # Fill missing numeric and demographic values.
        demo_missing = np.array(
            [
                not isinstance(val, (list, np.ndarray)) or len(val) == 0
                for val in df["demo_ratio"]
            ]
        )
        demo_ratio_vecs = [
            np.zeros(10) if missing else val
            for val, missing in zip(df["demo_ratio"], demo_missing)
        ]
        df["price_pct"] = df["price_pct"].fillna(0.0)
        price_pct = df["price_pct"].values

        desc_matrix = model.similarity(desc_vecs, desc_vecs).numpy()
        price_matrix = 1 - abs(price_pct[:, None] - price_pct[None, :])
        demo_ratio_matrix = js_divergence_matrix(np.array(demo_ratio_vecs))

        # Mask similarity for brand pairs without demographic ratios.
        demo_mask = demo_missing[:, None] | demo_missing[None, :]
        demo_ratio_matrix[demo_mask] = 0.0
        np.fill_diagonal(demo_ratio_matrix, 1.0)

        cat_group_matrix = cosine_similarity(cat_group_vecs)

        # Save similarity matrices.
        np.savez_compressed(
            output.path,
            brand_index=df["brand_name"].values,
            description=desc_matrix,
            price_pct=price_matrix,
            demo_ratio=demo_ratio_matrix,
            category_group=cat_group_matrix,
        )

        logging.info("[INFO] Saved content embeddings.")

    raw = pd.read_parquet(brand_data.path)
    df = raw.drop(columns="own_brand")
    compute_content_features(df)

    logging.info("[INFO] Processed content embeddings.")
