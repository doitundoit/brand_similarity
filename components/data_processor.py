from kfp.dsl import component, Output, Dataset


@component(
    base_image="us-west1-docker.pkg.dev/project-demo-498806/docker/brand-similarity:latest",
    install_kfp_package=False,
)
def get_data(reference_dt: str, brand_data: Output[Dataset], period: int = 90):
    """Fetch demo brand profile data and write a normalized parquet artifact."""

    from google.cloud import bigquery
    import pandas as pd
    import numpy as np
    import json

    SQL_BRAND = f"""
    SELECT  REPLACE(UPPER(src_entity_key), ' ', '') AS brand_name
    ,       COALESCE(src_entity_label, src_entity_key) AS brand_name_ko
    ,       src_term_group AS category_group
    ,       IF(LENGTH(src_text_profile) < 2, "", src_text_profile) AS description
    ,       src_numeric_score AS price_pct
    ,       src_distribution AS demo_ratio
    ,       src_is_new AS new_brand
    ,       src_is_owned AS own_brand
    ,       src_is_fashion AS fashion_brand
    FROM `project-demo-498806.sample.demo_brand_profiles`
    WHERE src_snapshot_dt >= PARSE_DATE('%Y%m%d', '{reference_dt}') - INTERVAL {period} DAY
    AND src_snapshot_dt <= PARSE_DATE('%Y%m%d', '{reference_dt}')
    QUALIFY src_snapshot_dt = MAX(src_snapshot_dt) OVER()
    """

    client = bigquery.Client(project="project-demo-498806")

    def parse_json_value(value, default):
        """Parse JSON-like values and fall back to a default on invalid input."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return default
        return value

    def normalize_yn(value) -> str:
        """Normalize truthy source values to a Y/N flag."""
        if isinstance(value, str):
            return "Y" if value.strip().upper() in {"Y", "YES", "TRUE", "1"} else "N"
        return "Y" if bool(value) else "N"

    def normalize_binary(value) -> int:
        """Normalize truthy source values to a binary integer flag."""
        if isinstance(value, str):
            return 1 if value.strip().upper() in {"Y", "YES", "TRUE", "1"} else 0
        return int(bool(value))

    def get_brand_data() -> pd.DataFrame:
        """Query demo brand data and normalize list-like and flag columns."""
        brand_df = client.query(SQL_BRAND).to_dataframe()
        brand_df.columns = brand_df.columns.str.lower()

        brand_df["category_group"] = brand_df["category_group"].map(
            lambda x: parse_json_value(x, [])
        )
        brand_df["demo_ratio"] = brand_df["demo_ratio"].map(
            lambda x: np.array(parse_json_value(x, []))
        )
        brand_df["new_brand"] = brand_df["new_brand"].map(normalize_yn)
        brand_df["own_brand"] = brand_df["own_brand"].map(normalize_binary)
        brand_df["fashion_brand"] = brand_df["fashion_brand"].map(normalize_binary)

        return brand_df

    merged = get_brand_data()

    # Exclude brands that have neither category terms nor demographic ratios.
    category_missing = merged["category_group"].apply(
        lambda x: len(x) == 0 if isinstance(x, (list, np.ndarray)) else pd.isna(x)
    )
    demo_missing = merged["demo_ratio"].apply(
        lambda x: len(x) == 0 if isinstance(x, (list, np.ndarray)) else pd.isna(x)
    )
    null_cond = category_missing & demo_missing

    brand_data_df = merged[~null_cond].copy()
    brand_data_df.to_parquet(brand_data.path)
