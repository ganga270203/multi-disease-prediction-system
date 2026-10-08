SYMPTOM_MAP = {
    "Diabetes": ["glucose", "sugar", "thirst", "frequent urination", "weight loss", "hunger"],
    "Heart Disease": ["chest pain", "shortness of breath", "angina", "fatigue", "palpitations"],
    "Parkinson's Disease": ["tremor", "stiffness", "balance", "speech", "shaking"],
    "Liver Disease": ["jaundice", "yellow skin", "abdominal pain", "swelling", "dark urine"],
    "Kidney Disease": ["swelling feet", "puffy eyes", "fatigue", "dry skin", "blood in urine"]
}

def analyze_symptoms(text):
    text_lower = text.lower()
    matched = []

    for disease, terms in SYMPTOM_MAP.items():
        found = [t for t in terms if t in text_lower]
        if found:
            matched.append({
                "name": disease,
                "matches": found,
                "confidence": min(len(found) * 35, 95)
            })

    matched = sorted(matched, key=lambda x: x["confidence"], reverse=True)
    return {
        "user_input": text,
        "possible_diseases": matched
    }

def format_bot_response(analysis):
    diseases = analysis.get("possible_diseases", [])
    if not diseases:
        return (
            "I could not clearly map your described symptoms to a specific condition profile. "
            "Please describe other symptoms you are experiencing, or select a condition from the Diagnostic Center."
        )

    response = "Based on your symptom description, here are relevant screening assessments:\n\n"
    for d in diseases:
        symptoms_str = ", ".join(d["matches"])
        response += f"- **{d['name']}** (Matched cues: _{symptoms_str}_)\n"

    response += "\n*Note: Use the buttons below or head to the Prediction Center to test specific clinical values.*"
    return response