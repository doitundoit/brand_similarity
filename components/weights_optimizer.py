from kfp.dsl import component, Artifact, Input
from typing import NamedTuple


@component(
    base_image="us-west1-docker.pkg.dev/project-demo-498806/docker/brand-similarity:latest",
    install_kfp_package=False,
)
def optimize_weights(
    interaction_embeddings: Input[Artifact],
    content_embeddings: Input[Artifact],
    dataset: str,
    reference_dt: str,
    project_path: str = "data/brand_similarity",
    max_trials: int = 50,
    ndcg_k: int = 10,
    period: int = 90,
) -> NamedTuple(
    "Outputs",
    [("content_weights", str), ("interaction_weights", str), ("global_weights", str)],
):
    """Optimize content and interaction feature weights with Vertex Vizier."""

    import io
    import numpy as np
    from datetime import datetime
    import logging
    import json
    import gcsfs
    from itertools import chain
    from sklearn.metrics.pairwise import cosine_similarity
    from google.cloud import bigquery
    from google.cloud.aiplatform.gapic import (
        VizierServiceClient,
        SuggestTrialsRequest,
        CompleteTrialRequest,
    )
    from utils import preprocess_matrix, map_brand_index

    def calculate_content_score(trial, content_data):
        """Apply trial content weights and return weighted content score matrices."""

        # read data
        data = content_data

        # Parse Weights
        feature_list = [key for key in data.keys() if key != "brand_index"]
        params = {p.parameter_id: p.value for p in trial.parameters}
        weights = {
            k: params.get(f"cw_{k}", 0.1) for k in feature_list
        }

        # Normalize weights using the same strategy as interaction weights.
        total_weight = sum(weights.values()) or 1.0

        # Apply feature weights.
        scores = {}
        for feature_name in weights.keys():
            matrix = data[feature_name]

            weight = weights.get(feature_name, 0.1)
            weighted_norm = weight / total_weight
            weighted_matrix = weighted_norm * matrix

            scores.update({feature_name: weighted_matrix})

        # Compute the aggregate content score.
        score = np.sum(list(scores.values()), axis=0)
        scores.update({"score": score})

        result = {"scores": scores, "brand_index": data["brand_index"]}
        return result

    def calculate_interaction_score(trial, interaction_data):
        """Apply trial interaction weights and return weighted interaction matrices."""

        # read data
        data = interaction_data

        # Parse Weights
        params = {p.parameter_id: p.value for p in trial.parameters}
        weights = {
            k: params.get(f"iw_{k}", 0.1) for k in data["feature_name"]
        }

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

        # Compute the aggregate interaction score.
        score = np.sum(list(scores.values()), axis=0)
        scores.update({"score": score})

        data = {"scores": scores, "brand_index": brand_indices}
        return data

    def dcg_at_k(r, k):
        """Calculate discounted cumulative gain at the requested cutoff."""
        r = np.asarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.0

    def ndcg_at_k(r, k):
        """Calculate normalized discounted cumulative gain at the requested cutoff."""
        dcg_max = dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.0
        return dcg_at_k(r, k) / dcg_max

    def calculate_ndcg(predictions, validation_df, k=10):
        """
        predictions: {brand_name: [list of predicted top brands]}
        validation_df: DataFrame [target_brand, rank, similar_brand]
        """
        ndcg_scores = []

        # Pre-process validation lookup
        validation_lookup = {}
        for target in validation_df["target_brand"].unique():
            sub = validation_df[validation_df["target_brand"] == target]
            validation_lookup[target] = {
                row["similar_brand"]: 1.0 / row["rank"] for _, row in sub.iterrows()
            }

        valid_targets = 0
        for target_brand, predicted_brands in predictions.items():
            if target_brand not in validation_lookup:
                continue

            ground_truth = validation_lookup[target_brand]
            relevance = [ground_truth.get(pb, 0.0) for pb in predicted_brands[:k]]
            score = ndcg_at_k(relevance, k)
            ndcg_scores.append(score)
            valid_targets += 1

        return np.mean(ndcg_scores) if valid_targets > 0 else 0.0

    def evaluate_metrics(trial, content, interaction, validation_data, k=10):
        """Evaluate a Vizier trial against validation rankings using NDCG."""

        # Parse Weights
        params = {p.parameter_id: p.value for p in trial.parameters}
        interaction_weight = params.get("interaction_weight", 0.5)
        content_weight = 1.0 - interaction_weight

        # Map brands shared by content and interaction artifacts.
        all_brands = sorted(
            set.intersection(
                set(content["brand_index"]), set(interaction["brand_index"])
            )
        )
        brand_to_idx = {brand_name: i for i, brand_name in enumerate(all_brands)}

        content_score = map_brand_index(data=content, brand_to_idx=brand_to_idx)
        interaction_score = map_brand_index(data=interaction, brand_to_idx=brand_to_idx)

        # Preprocess all-zero rows before cosine similarity.
        content_score = preprocess_matrix(content_score)
        interaction_score = preprocess_matrix(interaction_score)

        weighted_content = cosine_similarity(content_score) * content_weight
        weighted_interaction = cosine_similarity(interaction_score) * interaction_weight
        total_scores = weighted_content + weighted_interaction
        np.fill_diagonal(total_scores, 1.0)

        # Build top-K predictions for each brand.
        result = {}
        brand_indices = np.array(list(all_brands))
        for i, brand_name in enumerate(brand_indices):

            total_score = total_scores[i]
            top_idx = np.argsort(total_score)[::-1][1 : k + 1]
            result[brand_name] = brand_indices[top_idx]

        # evaluation
        eval_metrics = calculate_ndcg(result, validation_data, k)

        return eval_metrics

    def create_study(client, content_data, interaction_data, suffix: str = "_test"):
        """Create a Vizier study for content, interaction, and global weights."""

        # create study

        # weight names
        c_weights = [key for key in content_data.keys() if key != "brand_index"]
        i_weights = interaction_data["feature_name"]

        # Define parameters to optimize.
        params = []
        params.append(
            {
                "parameter_id": "interaction_weight",
                "double_value_spec": {"min_value": 0.0, "max_value": 0.5},
            }
        )

        for k in c_weights:
            params.append(
                {
                    "parameter_id": f"cw_{k}",
                    "double_value_spec": {"min_value": 0.0, "max_value": 1.0},
                }
            )

        for k in i_weights:
            params.append(
                {
                    "parameter_id": f"iw_{k}",
                    "double_value_spec": {"min_value": 0.0, "max_value": 1.0},
                }
            )

        study_spec = {
            "metrics": [{"metric_id": "ndcg_at_k", "goal": "MAXIMIZE"}],
            "parameters": params,
            "algorithm": "ALGORITHM_UNSPECIFIED",
        }

        parent = "projects/project-demo-498806/locations/us-west1"
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        display_name = f"brand_similarity{suffix}_{current_time}"

        study_params = {"display_name": display_name, "study_spec": study_spec}
        study = client.create_study(parent=parent, study=study_params)

        study_info = {
            "study_name": study.name,
            "display_name": display_name,
            "study_spec": study_spec,
            "study": study,
        }

        study_nums = study.name.split("/")[-1]
        study_url = f"https://console.cloud.google.com/vertex-ai/experiments/locations/us-west1/studies/{study_nums}?project=project-demo-498806"
        logging.info(f"[INFO] Study name: {study.name}")
        logging.info(f"[INFO] Display name: {display_name}")
        logging.info(f"[INFO] URL: {study_url}")

        return study_info

    def run_trials(
        client,
        study_info,
        content_data,
        interaction_data,
        validation_data,
        trial_idx,
        k: int = 10,
    ):
        """Run one suggested Vizier trial and report its NDCG measurement."""

        future = client.suggest_trials(
            request=SuggestTrialsRequest(
                parent=study_info.get("study_name"),
                suggestion_count=1,
            )
        )
        suggested_trials = future.result().trials

        # Use the single trial suggested by Vizier.
        trial = suggested_trials[0]
        trial_name = trial.name
        study_spec = study_info.get("study_spec")

        content = calculate_content_score(trial=trial, content_data=content_data)

        interaction = calculate_interaction_score(
            trial=trial, interaction_data=interaction_data
        )

        eval_metrics = evaluate_metrics(
            trial=trial,
            content=content,
            interaction=interaction,
            validation_data=validation_data,
            k=k,
        )

        # Report the evaluation result for this trial.
        client.add_trial_measurement(
            {
                "trial_name": trial_name,
                "measurement": {
                    "metrics": [
                        {
                            "metric_id": study_spec["metrics"][0]["metric_id"],
                            "value": eval_metrics,
                        }
                    ]
                },
            }
        )
        logging.info(f"[Trial {trial_idx+1}] NDCG@{k} -> {eval_metrics}")

        # Mark the Vizier trial as complete.
        client.complete_trial(request=CompleteTrialRequest(name=trial_name))

    def default_weights(content_data, interaction_data):
        """Return neutral fallback weights for all available features."""
        content_features = [key for key in content_data.keys() if key != "brand_index"]
        interaction_features = interaction_data["feature_name"].tolist()
        return (
            {feature: 1.0 for feature in content_features},
            {feature: 1.0 for feature in interaction_features},
            {"interaction_weight": 0.5, "content_weight": 0.5},
        )

    # main
    api_endpoint = "us-west1-aiplatform.googleapis.com"
    client = VizierServiceClient(client_options=dict(api_endpoint=api_endpoint))
    fs = gcsfs.GCSFileSystem()

    with fs.open(f"{content_embeddings.uri}.npz", "rb") as file:
        content_data = np.load(io.BytesIO(file.read()), allow_pickle=True)
    with fs.open(f"{interaction_embeddings.uri}.npz", "rb") as file:
        interaction_data = np.load(io.BytesIO(file.read()), allow_pickle=True)

    c_weights, i_weights, g_weights = default_weights(content_data, interaction_data)

    if max_trials <= 0:
        logging.info("[INFO] Vizier skipped because max_trials <= 0.")
        return (
            json.dumps(c_weights, ensure_ascii=False, indent=2),
            json.dumps(i_weights, ensure_ascii=False, indent=2),
            json.dumps(g_weights, ensure_ascii=False, indent=2),
        )

    SQL_VALIDATION = """
    SELECT  src_target_key AS TARGET_BRAND
    ,       src_candidate_key AS SIMILAR_BRAND
    ,       CAST(src_rank_label AS FLOAT64) AS RANK
    ,       src_created_at AS CREATE_DT
    FROM `project-demo-498806.sample.demo_brand_similarity_validation`
    WHERE src_candidate_key IS NOT NULL
    QUALIFY src_created_at = MAX(src_created_at) OVER()
    """

    bq_client = bigquery.Client(project="project-demo-498806")
    validation_data = bq_client.query(SQL_VALIDATION).to_dataframe()
    validation_data.columns = validation_data.columns.str.lower()

    if validation_data.empty:
        logging.info("[INFO] Vizier skipped because validation data is empty.")
        return (
            json.dumps(c_weights, ensure_ascii=False, indent=2),
            json.dumps(i_weights, ensure_ascii=False, indent=2),
            json.dumps(g_weights, ensure_ascii=False, indent=2),
        )

    suffix = "_1y" if period == 365 else ""
    suffix += "_test" if "_test" in dataset else ""

    study_info = create_study(
        client=client,
        suffix=suffix,
        content_data=content_data,
        interaction_data=interaction_data,
    )

    for i in range(max_trials):
        run_trials(
            client=client,
            study_info=study_info,
            content_data=content_data,
            interaction_data=interaction_data,
            validation_data=validation_data,
            trial_idx=i,
            k=ndcg_k,
        )

    logging.info("[INFO] All trials finished.")
    STUDY_NAME = study_info.get("study_name")

    logging.info(f"[INFO] Vizier study: {STUDY_NAME}")

    # Read the best weights from the completed Vizier study.
    optimal_trials = client.list_optimal_trials(parent=STUDY_NAME).optimal_trials
    if not optimal_trials:
        logging.info("[INFO] Vizier returned no optimal trials; using default weights.")
        return (
            json.dumps(c_weights, ensure_ascii=False, indent=2),
            json.dumps(i_weights, ensure_ascii=False, indent=2),
            json.dumps(g_weights, ensure_ascii=False, indent=2),
        )

    best = optimal_trials[0]

    with fs.open(f"{content_embeddings.uri}.npz", "rb") as file:
        content_data = np.load(io.BytesIO(file.read()), allow_pickle=True)
    content_features = {key for key in content_data.keys() if key != "brand_index"}

    c_weights = {}
    i_weights = {}
    g_weights = {}

    for p in best.parameters:

        feature_name = p.parameter_id
        weight = p.value

        if feature_name.startswith("cw_"):
            name = feature_name[3:]
            if name in content_features:
                c_weights.update({name: weight})

        elif feature_name.startswith("iw_"):
            name = feature_name[3:]
            i_weights.update({name: weight})

        else:
            g_weights.update({feature_name: weight})
            g_weights.update({"content_weight": 1 - weight})

    content_weights = json.dumps(c_weights, ensure_ascii=False, indent=2)
    interaction_weights = json.dumps(i_weights, ensure_ascii=False, indent=2)
    global_weights = json.dumps(g_weights, ensure_ascii=False, indent=2)

    return (content_weights, interaction_weights, global_weights)
