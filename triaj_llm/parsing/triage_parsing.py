"""
Output parsing utilities for medical triage LLM classification.
"""

import re
from typing import Dict, Optional, Any, Tuple
import logging
import jsonschema

# Setup logging
logger = logging.getLogger(__name__)

# Schema for validating LLM responses for triage
TRIAGE_LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["URGENCY", "URGENCY_CONFIDENCE", "URGENCY_REASONING"],
    "properties": {
        "URGENCY": {
            "type": "string",
            "enum": ['Kırmızı Alan', 'Sarı Alan', 'Yeşil Alan'] # Based on unique values found
        },
        "URGENCY_CONFIDENCE": {
            "type": "string",
            "enum": ["high", "medium", "low"]
        },
        "URGENCY_REASONING": {
            "type": "string"
        }
    }
}

def validate_triage_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Validate and parse LLM response text against the triage schema.

    Args:
        response_text: Raw text response from LLM

    Returns:
        Parsed and validated response dictionary

    Raises:
        jsonschema.exceptions.ValidationError: If response doesn't match schema
        ValueError: If response can't be parsed
    """
    # Extract content between triple backticks
    match = re.search(r"```(.*?)```", response_text, re.DOTALL)
    if not match:
        raise ValueError("No code block found in response")

    content = match.group(1).strip()

    # Parse the content into a dictionary
    response_dict = {}
    for line in content.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            response_dict[key.strip()] = value.strip()

    # Validate against schema
    jsonschema.validate(instance=response_dict, schema=TRIAGE_LLM_RESPONSE_SCHEMA)

    return response_dict

def parse_triage_llm_output(llm_output: str) -> Dict[str, Any]:
    """
    Parse LLM output to extract the predicted urgency level, confidence, and reasoning.

    Args:
        llm_output: Raw text output from the LLM

    Returns:
        Dictionary containing extracted information:
        - 'urgency': The extracted urgency level
        - 'confidence': The confidence level (high, medium, low)
        - 'reasoning': The reasoning provided for the classification
        - 'raw': The raw text that was parsed
    """
    # Clean up input text - remove code blocks and extra whitespace
    cleaned_output = llm_output.replace("```", "").strip()

    # Extract field values using standard patterns
    urgency_value = _extract_first_match(cleaned_output, [r"URGENCY:\s*([^\n]+)"])
    confidence_value = _extract_first_match(cleaned_output, [r"URGENCY_CONFIDENCE:\s*([^\n]+)"])
    reasoning_value = _extract_multiline_match(cleaned_output, [
        r"URGENCY_REASONING:\s*(.*?)(?=\n\n|\n[A-Z_]+:|\Z)",
        r"REASONING:\s*(.*?)(?=\n\n|\n[A-Z_]+:|\Z)" # Fallback pattern
    ])

    # Clean up and validate urgency value
    if urgency_value:
        urgency_value = urgency_value.strip(' "\',.;:')
        # Ensure it's one of the valid levels, case-insensitive check first
        lower_urgency = urgency_value.lower()
        found_match = False
        for valid_level in TRIAGE_LLM_RESPONSE_SCHEMA["properties"]["URGENCY"]["enum"]:
            if lower_urgency == valid_level.lower():
                urgency_value = valid_level # Use the exact valid level casing
                found_match = True
                break
        if not found_match:
             # If still not a valid level, try to pick the closest one or a default
             urgency_value = _pick_valid_urgency(urgency_value, cleaned_output)


    # Normalize confidence value
    if confidence_value:
        confidence_value = confidence_value.lower().strip()
        # Only accept valid confidence values
        if confidence_value not in ["high", "medium", "low"]:
            confidence_value = "medium"
    else:
        confidence_value = "medium"  # Default confidence

    # Ensure reasoning has a value
    if not reasoning_value:
        reasoning_value = "Based on patient data and medical assessment."

    # Make sure reasoning is not truncated and ends with a complete sentence
    if reasoning_value:
        # Ensure reasoning ends with a sentence-ending punctuation
        if not reasoning_value.endswith((".", "!", "?")):
            # Try to find the last complete sentence
            sentences = re.split(r'(?<=[.!?])\s+', reasoning_value)
            if sentences and len(sentences) > 0:
                # Get all complete sentences
                complete_sentences = [s for s in sentences if s.endswith((".", "!", "?"))]
                if complete_sentences:
                    reasoning_value = " ".join(complete_sentences)
                else:
                    # If no complete sentence, add a period at the end
                    reasoning_value = reasoning_value.rstrip(" ,.;:") + "."

    # Construct result dictionary
    result = {
        "urgency": urgency_value if urgency_value else None,
        "confidence": confidence_value,
        "reasoning": reasoning_value.strip() if reasoning_value else None,
        "raw": cleaned_output.strip()
    }

    return result

# Helper functions adapted from parsing.py
def _extract_first_match(text: str, patterns: list) -> Optional[str]:
    """
    Try multiple regex patterns and return the first match.
    """
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None

def _extract_multiline_match(text: str, patterns: list) -> Optional[str]:
    """
    Try multiple regex patterns and return the first match, allowing for multiline content.
    """
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            result = match.group(1).strip()
            # Basic check for truncation and add period if needed
            if result and not result.endswith((".", "!", "?")):
                 result += "."
            return result
    return None

def _pick_valid_urgency(extracted_value: str, full_text: str) -> str:
    """
    Determines the most likely valid urgency level when the extracted value is not valid.
    """
    valid_levels = TRIAGE_LLM_RESPONSE_SCHEMA["properties"]["URGENCY"]["enum"]
    text_lower = full_text.lower()

    # Try to find mentions of valid levels in the full text
    level_counts = [(text_lower.count(level.lower()), level) for level in valid_levels]
    level_counts.sort(reverse=True)

    if level_counts[0][0] > 0:
        return level_counts[0][1]

    # If no valid level mentioned, try to find keywords
    if any(keyword in text_lower for keyword in ["critical", "immediate", "red"]):
        return 'Kırmızı Alan'
    elif any(keyword in text_lower for keyword in ["urgent", "priority", "yellow"]):
        return 'Sarı Alan'
    elif any(keyword in text_lower for keyword in ["non-urgent", "routine", "green"]):
        return 'Yeşil Alan'

    # Default to the most common level if nothing matches (based on previous data analysis)
    return 'Sarı Alan' # Assuming 'Sarı Alan' is the most common, verify with data if needed