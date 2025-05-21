from typing import Dict, Any, Optional

def format_triage_data_for_llm(
    patient_data: Dict[str, Any],
    feature_descriptions: Optional[Dict[str, str]] = None,
    format_type: str = "natural",
    include_label: bool = True
) -> str:
    """
    Format medical triage data as text for LLM processing.

    Args:
        patient_data: Dictionary of patient feature values (a row from the DataFrame).
        feature_descriptions: Dictionary mapping feature names to descriptions (optional).
        format_type: Type of formatting ('natural' or 'list').
        include_label: Whether to include the urgency label in the formatted output (for training/few-shot).

    Returns:
        Text representation of patient data.
    """
    if format_type == "list":
        # Simple list format
        formatted_text = "\n".join([f"{feat}: {value}" for feat, value in patient_data.items()])

    elif format_type == "natural":
        # More natural language description
        if feature_descriptions is None:
            # Generic descriptions if not provided
            feature_descriptions = {
                feat: feat.replace("_", " ") for feat in patient_data.keys()
            }

        result = "Patient data: "

        # Include label if requested
        if include_label:
            urgency_label = patient_data.get("doğru triyaj", None)
            if urgency_label is not None:
                result += f"This patient was triaged to the '{urgency_label}' area. "

        # Add relevant medical features
        age = patient_data.get("yaş", None)
        if age is not None:
            result += f"Age: {age}. "

        gender = patient_data.get("cinsiyet", None)
        if gender is not None:
            result += f"Gender: {gender}. "

        systolic_bp = patient_data.get("sistolik kb", None)
        diastolic_bp = patient_data.get("diastolik kb", None)
        if systolic_bp is not None and diastolic_bp is not None:
            result += f"Blood Pressure: {systolic_bp}/{diastolic_bp} mmHg. "
        elif systolic_bp is not None:
             result += f"Systolic Blood Pressure: {systolic_bp} mmHg. "
        elif diastolic_bp is not None:
             result += f"Diastolic Blood Pressure: {diastolik_bp} mmHg. "

        respiration_rate = patient_data.get("solunum sayısı", None)
        if respiration_rate is not None:
            result += f"Respiration Rate: {respiration_rate}. "

        pulse = patient_data.get("nabız", None)
        if pulse is not None:
            result += f"Pulse: {pulse}. "

        fever = patient_data.get("ateş", None)
        if fever is not None:
            result += f"Fever: {fever}°C. " # Assuming Celsius

        saturation = patient_data.get("saturasyon", None)
        if saturation is not None:
            result += f"Oxygen Saturation: {saturation}%. "

        additional_conditions = patient_data.get("ek hastalıklar", None)
        if additional_conditions and additional_conditions != "Unknown":
             result += f"Additional conditions: {additional_conditions}. "

        # Include symptom information - need to decide how to best represent these
        # For now, just include the general symptom columns if they are not "Unknown"
        symptoms_01 = patient_data.get("semptomlar_non travma_genel 01", None)
        if symptoms_01 and symptoms_01 != "Unknown":
            result += f"General symptoms: {symptoms_01}. "

        symptoms_02 = patient_data.get("semptomlar_non travma_genel 02", None)
        if symptoms_02 and symptoms_02 != "Unknown":
            result += f"Other symptoms: {symptoms_02}. "

        # Add other relevant columns as needed, potentially grouping them
        # For example, trauma types, specific disease categories, etc.
        # This part can be expanded based on which features are most informative for triage.

        # Catch any remaining features using their generic descriptions if not already covered
        # This might be too verbose, consider carefully which features to include.
        # for feat, value in patient_data.items():
        #     if feat not in ["doğru triyaj", "yaş", "cinsiyet", "sistolik kb", "diastolik kb",
        #                     "solunum sayısı", "nabız", "ateş", "saturasyon", "ek hastalıklar",
        #                     "semptomlar_non travma_genel 01", "semptomlar_non travma_genel 02"] and value is not None:
        #         desc = feature_descriptions.get(feat, feat.replace("_", " "))
        #         result += f"{desc}: {value}. "

        formatted_text = result.strip()

    else:
        raise ValueError(f"Unknown format type: {format_type}")

    return formatted_text

# You might also want a batch formatting function similar to batch_format_flows
# def batch_format_triage_data(...):
#    pass