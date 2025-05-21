# Plan for Adapting LLM Classification Framework for Medical Triage

This document outlines the plan to adapt the provided LLM classification framework for predicting medical urgency using the triage data.

## Objective

Adapt the full LLM classification framework to predict medical urgency using the processed triage data (`data/output/processed_triage.parquet`).

## Detailed Steps

1.  **Analyze the Medical Triage Data Structure:**
    Re-examine the structure of the processed medical data in `data/output/processed_triage.parquet` to understand available features and confirm the name and format of the target urgency label column ('doğru triyaj').

2.  **Define Medical Urgency Output Classes:**
    Based on the values found in the 'doğru triyaj' column (e.g., 'Kırmızı Alan', 'Sarı Alan', 'Yeşil Alan'), explicitly define the valid output classes that the LLM should predict.

3.  **Adapt Data Loading:**
    Modify the `import_data` method within the classification framework (or create a new data loading function) to specifically load the `data/output/processed_triage.parquet` file.

4.  **Adapt Data Formatting for LLM Prompts:**
    Create a new function (or significantly modify the existing `format_flow_for_llm`) to transform the medical patient data (features like age, vital signs, reported symptoms, existing conditions) into a clear, structured text format suitable for an LLM for triage.

5.  **Develop New Prompt Templates for Triage:**
    Write new prompt templates, similar to `prompt_templates.py`, tailored for medical urgency classification. These templates will instruct the LLM, provide valid urgency levels as output constraints, and define the required output format. Adapt "no_shot", "few_shot", and "multi_task" approaches for the triage context.

6.  **Adapt LLM Output Parsing Logic:**
    Modify the `parse_llm_output` function and update the `LLM_RESPONSE_SCHEMA` to correctly extract the predicted medical urgency level, its confidence, and the LLM's reasoning from the API response. Change expected keys from 'APP' and 'DEVICE' to 'URGENCY' and relevant triage fields.

7.  **Adapt the Classification Class:**
    Create a new Python class (e.g., `TriageLLMClassifier`) or heavily modify the existing `TrafficClassifier` to incorporate the new medical data loading, formatting, and parsing logic. Update methods like `pick_random_flow` and `classify` to handle the medical dataset and the 'doğru triyaj' label.

8.  **Adapt Evaluation Logic:**
    Modify the `evaluate_classifier` function to calculate accuracy and other relevant metrics based on the comparison between the LLM's predicted urgency levels and the actual 'doğru triyaj' labels.

9.  **Integrate into Project Structure:**
    Create a new directory (e.g., `triaj_llm/`) within the project to house the adapted classification code, prompt templates, and parsing logic, keeping it separate from the original network traffic code.

10. **Refine and Test:**
    Thoroughly test the entire pipeline to ensure correct processing, prediction, and evaluation. Refine prompts and code based on results.

## Process Diagram

```mermaid
graph TD
    A[data/triaj_data.csv] --> B(data/triage_data_processor.py);
    B --> C[data/output/processed_triage.parquet];
    C --> D(Triage LLM Classifier);
    D --> E(Data Formatting for Triage LLM);
    D --> F(Prompt Templates for Triage);
    D --> G(LLM Provider);
    E & F & G --> H(LLM API Call);
    H --> I(LLM Response);
    I --> J(Parsing Triage LLM Output);
    J --> K(Triage Classification Result);
    K --> L(Triage Evaluation Logic);
    L --> M[Triage Evaluation Report];