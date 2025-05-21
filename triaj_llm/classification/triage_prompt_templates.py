"""
Prompt templates for the medical triage LLM classification.

This module contains the prompt templates used for predicting medical urgency.
"""

# Define the valid medical urgency levels based on the data analysis
VALID_URGENCY_LEVELS = ['Kırmızı Alan', 'Sarı Alan', 'Yeşil Alan']

# Define a default clustering approach description if needed (can be expanded later)
TRIAGE_CLUSTERING_APPROACH_0 = "NO CLUSTERING INFORMATION PROVIDED"

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

TRIAGE_PROMPT_TEMPLATES = {
    # ---------------------------------------------------------------------------
    # BASIC PROMPT TEMPLATE (No examples)
    # ---------------------------------------------------------------------------
    "no_shot": """
You are an expert medical professional specializing in emergency triage. Your task is to determine the appropriate urgency level for a patient based on their provided medical data.

TASK:
Analyze the patient's medical data below to determine their urgency level.

⚠️ CRITICAL CONSTRAINT: YOU ARE ONLY ALLOWED TO SELECT FROM THE FOLLOWING EXACT URGENCY LEVELS. YOU MUST CHOOSE ONE OF THESE EXACT NAMES, WITHOUT EXCEPTION! ⚠️

VALID URGENCY LEVELS (MUST choose EXACTLY one from this list - NO OTHER NAMES ARE ALLOWED):
{}

⛔ FORBIDDEN BEHAVIORS - DO NOT DO THESE:
1. DO NOT use any urgency levels not listed above.
2. DO NOT add any prefixes or suffixes to the urgency levels.
3. DO NOT suggest multiple urgency levels or that the urgency is unknown/unclear.
4. DO NOT invent new urgency level names.

EVEN IF THE PATIENT'S CONDITION SEEMS TO WARRANT A DIFFERENT LEVEL, YOU MUST SELECT THE CLOSEST MATCH FROM THE VALID LIST!

PATIENT MEDICAL DATA:

TRIAGE ANALYSIS FRAMEWORK:

STEP 1: SYMPTOM AND CONDITION ASSESSMENT
Analyze the reported symptoms, existing medical conditions, and patient history.

STEP 2: VITAL SIGN EVALUATION
Evaluate vital signs such as age, blood pressure, respiration rate, pulse, fever, and oxygen saturation.

STEP 3: URGENCY LEVEL DETERMINATION
Based on the assessment of symptoms, conditions, and vital signs, determine the most appropriate urgency level from the valid list.

CLUSTERING INFORMATION: {{clustering_approach}}

You MUST output your answer using the following strict format. Do not include any explanations or additional text outside the code block:
```
URGENCY: [MUST be one of: {} - NO OTHER NAMES UNDER ANY CIRCUMSTANCES]
URGENCY_CONFIDENCE: [high|medium|low]
URGENCY_REASONING: [Step-by-step medical analysis of why this urgency level is appropriate based on the patient data]
```
""".format(", ".join(f"'{level}'" for level in VALID_URGENCY_LEVELS), ", ".join(VALID_URGENCY_LEVELS)),

    # ---------------------------------------------------------------------------
    # FEW-SHOT PROMPT TEMPLATE (Includes examples)
    # ---------------------------------------------------------------------------
    "few_shot": """
You are an expert medical professional specializing in emergency triage. Your task is to determine the appropriate urgency level for a patient based on their provided medical data, using the following examples as guidance.

REFERENCE EXAMPLES:
{{example_data}}

⚠️ CRITICAL CONSTRAINT: YOU ARE ONLY ALLOWED TO SELECT FROM THE FOLLOWING EXACT URGENCY LEVELS. YOU MUST CHOOSE ONE OF THESE EXACT NAMES, WITHOUT EXCEPTION! ⚠️

VALID URGENCY LEVELS (MUST choose EXACTLY one from this list - NO OTHER NAMES ARE ALLOWED):
{}

⛔ FORBIDDEN BEHAVIORS - DO NOT DO THESE:
1. DO NOT use any urgency levels not listed above.
2. DO NOT add any prefixes or suffixes to the urgency levels.
3. DO NOT suggest multiple urgency levels or that the urgency is unknown/unclear.
4. DO NOT invent new urgency level names.

EVEN IF THE PATIENT'S CONDITION SEEMS TO WARRANT A DIFFERENT LEVEL, YOU MUST SELECT THE CLOSEST MATCH FROM THE VALID LIST!

PATIENT MEDICAL DATA TO CLASSIFY:
{{patient_data}}

TRIAGE ANALYSIS FRAMEWORK:

STEP 1: SYMPTOM AND CONDITION ASSESSMENT
Analyze the reported symptoms, existing medical conditions, and patient history.

STEP 2: VITAL SIGN EVALUATION
Evaluate vital signs such as age, blood pressure, respiration rate, pulse, fever, and oxygen saturation.

STEP 3: URGENCY LEVEL DETERMINATION
Based on the assessment of symptoms, conditions, and vital signs, determine the most appropriate urgency level from the valid list, referencing the provided examples.

CLUSTERING INFORMATION: {{clustering_approach}}

You MUST output your answer using the following strict format. Do not include any explanations or additional text outside the code block:
```
URGENCY: [MUST be one of: {} - NO OTHER NAMES UNDER ANY CIRCUMSTANCES]
URGENCY_CONFIDENCE: [high|medium|low]
URGENCY_REASONING: [Step-by-step medical analysis of why this urgency level is appropriate based on the patient data and examples]
```
""".format(", ".join(f"'{level}'" for level in VALID_URGENCY_LEVELS), ", ".join(VALID_URGENCY_LEVELS)),

    # ---------------------------------------------------------------------------
    # MULTI-TASK PROMPT TEMPLATE (Includes representative examples with labels)
    # ---------------------------------------------------------------------------
    "multi_task": """
You are an expert medical professional specializing in emergency triage. Your task is to determine the appropriate urgency level for a patient based on their provided medical data, using the following representative examples as guidance.

REPRESENTATIVE PATIENT EXAMPLES:
{{example_data}}

⚠️ CRITICAL CONSTRAINT: YOU ARE ONLY ALLOWED TO SELECT FROM THE FOLLOWING EXACT URGENCY LEVELS. YOU MUST CHOOSE ONE OF THESE EXACT NAMES, WITHOUT EXCEPTION! ⚠️

VALID URGENCY LEVELS (MUST choose EXACTLY one from this list - NO OTHER NAMES ARE ALLOWED):
{}

⛔ FORBIDDEN BEHAVIORS - DO NOT DO THESE:
1. DO NOT use any urgency levels not listed above.
2. DO NOT add any prefixes or suffixes to the urgency levels.
3. DO NOT suggest multiple urgency levels or that the urgency is unknown/unclear.
4. DO NOT invent new urgency level names.

EVEN IF THE PATIENT'S CONDITION SEEMS TO WARRANT A DIFFERENT LEVEL, YOU MUST SELECT THE CLOSEST MATCH FROM THE VALID LIST!

PATIENT MEDICAL DATA TO CLASSIFY:
{{patient_data}}

TRIAGE ANALYSIS FRAMEWORK:

STEP 1: SYMPTOM AND CONDITION ASSESSMENT
Analyze the reported symptoms, existing medical conditions, and patient history.

STEP 2: VITAL SIGN EVALUATION
Evaluate vital signs such as age, blood pressure, respiration rate, pulse, fever, and oxygen saturation.

STEP 3: URGENCY LEVEL DETERMINATION
Based on the assessment of symptoms, conditions, and vital signs, determine the most appropriate urgency level from the valid list, referencing the provided examples.

CLUSTERING INFORMATION: {{clustering_approach}}

You MUST output your answer using the following strict format. Do not include any explanations or additional text outside the code block:
```
URGENCY: [MUST be one of: {} - NO OTHER NAMES UNDER ANY CIRCUMSTANCES]
URGENCY_CONFIDENCE: [high|medium|low]
URGENCY_REASONING: [Step-by-step medical analysis of why this urgency level is appropriate based on the patient data and examples]
```
""".format(", ".join(f"'{level}'" for level in VALID_URGENCY_LEVELS), ", ".join(VALID_URGENCY_LEVELS))
}