from kfp import compiler, dsl

from components.data_processor import get_data
from components.contents_processor import process_content_embeddings
from components.interaction_processor import process_interaction_embeddings
from components.weights_optimizer import optimize_weights
from components.similarity_calculator import calculate_similarity
from components.result_uploader import update_result


@dsl.pipeline(
    name="brand_similarity",
    description="Calculate brand similarity",
)
def pipeline(
    dataset: str = "sample",
    reference_dt: str = "20260101",
    max_trials: int = 5,
    top_n: int = 100,
    period: int = 365,
):
    """Define the Kubeflow pipeline for brand similarity calculation."""

    brand_data = get_data(reference_dt=reference_dt, period=period)

    content_embeddings = process_content_embeddings(
        brand_data=brand_data.outputs["brand_data"]
    )
    interaction_embeddings = process_interaction_embeddings(
        reference_dt=reference_dt,
        brand_data=brand_data.outputs["brand_data"],
        period=period,
    )
    optimized_weights = optimize_weights(
        dataset=dataset,
        interaction_embeddings=interaction_embeddings.outputs["output"],
        content_embeddings=content_embeddings.outputs["output"],
        max_trials=max_trials,
        reference_dt=reference_dt,
        period=period,
    )
    result = calculate_similarity(
        content_weights=optimized_weights.outputs["content_weights"],
        interaction_weights=optimized_weights.outputs["interaction_weights"],
        global_weights=optimized_weights.outputs["global_weights"],
        interaction_embeddings=interaction_embeddings.outputs["output"],
        content_embeddings=content_embeddings.outputs["output"],
        brand_data=brand_data.outputs["brand_data"],
        reference_dt=reference_dt,
        top_n=top_n,
    )
    update_result(
        dataset=dataset,
        reference_dt=reference_dt,
        input_data=result.outputs["output"],
        period=period,
    )


def compile_pipeline():
    """Compile the brand similarity pipeline to a JSON package."""
    compiler.Compiler().compile(pipeline_func=pipeline, package_path="pipeline.json")


if __name__ == "__main__":
    compile_pipeline()
