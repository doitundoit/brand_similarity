from kfp.dsl import component, Input, Dataset


@component(
    base_image="us-west1-docker.pkg.dev/project-demo-498806/docker/brand-similarity:latest",
    install_kfp_package=False,
)
def update_result(
    dataset: str,
    reference_dt: str,
    input_data: Input[Dataset],
    table: str = "brand_similarity",
    period: int = 90,
):
    """Upload similarity results to BigQuery and refresh latest-result views."""

    from google.cloud import bigquery
    from google.api_core import exceptions
    from datetime import datetime, timedelta

    project = "project-demo-498806"
    table_name = f"{project}.{dataset}.{table}"

    if period == 365:
        table_name += "_1y"

    view_latest = table_name + "_latest"
    view_unnested = table_name + "_unnested"

    reference_date = datetime.strptime(reference_dt, "%Y%m%d")
    reference_dt_1y = (reference_date - timedelta(days=365)).strftime("%Y%m%d")

    client = bigquery.Client(project=project)

    # Delete existing rows after confirming the table exists.
    try:
        client.get_table(table_name)
        delete_sql = f"DELETE FROM {table_name} WHERE DT = '{reference_dt}' OR DT <= '{reference_dt_1y}'"
        client.query(delete_sql).result()

    except exceptions.NotFound:
        pass

    # Load parquet output into BigQuery.
    try:
        job_config = bigquery.LoadJobConfig(
            # autodetect=True,  # Enable schema autodetection.
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema_update_options=[
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
            ],
        )

        load_job = client.load_table_from_uri(
            source_uris=input_data.uri,
            destination=table_name,
            job_config=job_config,
            project=project,
        )

        load_job.result()

    except Exception as e:
        raise Exception(f"Error occurred while updating table: {str(e)}")

    # create latest view
    try:
        sql = f"""
        CREATE OR REPLACE VIEW {view_latest} AS
        SELECT *
        FROM {table_name}
        WHERE DT = '{reference_dt}'
        """
        client.query(sql).result()

    except Exception as e:
        raise Exception(f"Error occurred while creating {view_latest}: {str(e)}")

    # create unnested view
    try:
        sql = f"""
        CREATE OR REPLACE VIEW {view_unnested} AS
        WITH FLATTENED AS (
            SELECT  BRAND_NAME
            ,       BRAND_NAME_KO
            ,       SIMILARITY_TYPE
            ,       CAST(JSON_VALUE(sim, '$.rank') AS INT64) AS SIMILAR_RANK
            ,       JSON_VALUE(sim, '$.brand_name') AS SIMILAR_BRAND
            ,       CAST(JSON_VALUE(sim, '$.total_score') AS FLOAT64) AS SIMILAR_SCORE
            FROM {view_latest}, 
            UNNEST(JSON_EXTRACT_ARRAY(similarity)) AS sim
        )

        SELECT  F.BRAND_NAME
        ,       F.BRAND_NAME_KO
        ,       F.SIMILAR_RANK
        ,       F.SIMILAR_BRAND
        ,       M.BRAND_NAME_KO AS SIMILAR_BRAND_KO
        ,       ROUND(F.SIMILAR_SCORE, 4) AS SIMILAR_SCORE
        ,       F.SIMILARITY_TYPE
        FROM FLATTENED F
        LEFT JOIN {view_latest} M 
        ON F.SIMILAR_BRAND = M.BRAND_NAME 
        AND F.SIMILARITY_TYPE = M.SIMILARITY_TYPE
        """
        client.query(sql).result()

    except Exception as e:
        raise Exception(f"Error occurred while creating {view_unnested}: {str(e)}")
