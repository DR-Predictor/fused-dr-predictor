# risk_mapper.py
def map_risk(pred_index: int) -> str:
    if pred_index in [0, 1]:
        return "Risk: Low. Recommended Follow-up: 24 Months."
    elif pred_index == 2:
        return "Risk: Moderate. Recommended Follow-up: 12 Months."
    elif pred_index in [3, 4]:
        return "Risk: High. Recommended Follow-up: 3–6 Months. Refer to a specialist."