"""
Script to run evaluation experiments for the Triage LLM Classifier.
"""

import os
import logging
from triaj_llm.classification.triage_classifier import TriageLLMClassifier
from triaj_llm.classification.triage_prompt_templates import TRIAGE_CLUSTERING_APPROACH_0 # Import default clustering approach

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_evaluation(
    provider: str,
    model: str,
    prompt_type: str,
    num_cases: int = 10,
    batch_size: int = 1,
    rate_limit_delay: float = 2.0,
    save_results: bool = True,
    results_dir: str = "triaj_llm_results",
    sample_data_length: int = 10,
    clustering_approach: str = TRIAGE_CLUSTERING_APPROACH_0
):
    """
    Runs a single evaluation experiment for the Triage LLM Classifier.

    Args:
        provider: LLM provider name.
        model: Model name.
        prompt_type: Type of prompt to use.
        num_cases: Number of patient cases to classify.
        batch_size: Size of batches for classification.
        rate_limit_delay: Delay between API calls/batches.
        save_results: Whether to save results to a file.
        results_dir: Directory to save results in.
        sample_data_length: Number of sample patient cases for few-shot/multi-task prompts.
        clustering_approach: Description of clustering information to include in prompt.

    Returns:
        Dictionary with evaluation metrics.
    """
    try:
        logger.info(f"Initializing TriageLLMClassifier for experiment: Provider={provider}, Model={model}, Prompt Type={prompt_type}")
        classifier = TriageLLMClassifier(
            provider_name=provider,
            model=model,
            prompt_type=prompt_type,
            sample_data_length=sample_data_length,
            clustering_approach=clustering_approach
            # file_path is defaulted in the class
            # label_column is defaulted in the class
        )

        logger.info(f"Running evaluation experiment with {num_cases} cases...")
        evaluation_metrics = classifier.run_evaluation_experiment(
            num_cases=num_cases,
            batch_size=batch_size,
            rate_limit_delay=rate_limit_delay,
            save_results=save_results,
            results_dir=results_dir
        )

        print("\n===== Triage LLM Evaluation Summary =====")
        if "error" in evaluation_metrics:
            print(f"Experiment failed: {evaluation_metrics['error']}")
        else:
            print(f"Number of cases evaluated: {evaluation_metrics['num_evaluated']}")
            print(f"Errors encountered during classification: {evaluation_metrics['error_count']}")
            print(f"Urgency classification accuracy: {evaluation_metrics['urgency_accuracy']:.2%}")
            print(f"Confidence counts: {evaluation_metrics['confidence_counts']}")
            print(f"Detailed results saved to '{results_dir}' directory.")

        return evaluation_metrics

    except Exception as e:
        logger.error(f"An error occurred during the evaluation run: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    # Example usage: Configure and run an experiment
    # !!! IMPORTANT: Replace with your actual provider, model, and API key setup !!!
    # Ensure you have the necessary environment variables set or pass the api_key directly.

    # Example using Gemini (requires GOOGLE_API_KEY environment variable or api_key param)
    run_evaluation(
        provider="gemini",
        model="gemini-1.5-flash", # Or another suitable Gemini model
        prompt_type="no_shot", # Choose from "no_shot", "few_shot", "multi_task"
        num_cases=10, # Number of cases to test
        sample_data_length=5, # Number of examples for few-shot/multi-task
        rate_limit_delay=1.0 # Adjust based on rate limits
    )

    # You can add more calls to run_evaluation with different configurations
    # to compare different models, prompt types, etc.
    # Example:
    # run_evaluation(
    #     provider="gemini",
    #     model="gemini-1.5-flash",
    #     prompt_type="few_shot",
    #     num_cases=20,
    #     sample_data_length=10,
    #     rate_limit_delay=1.5
    # )