def sort_dict(dico: dict) -> dict:
    return dict(sorted(dico.items(), key=lambda a: a[1]))


influent_state_col = sort_dict(
    {
        "time": 0,  # Day
        "D": 1,  # Day-1
        "S1": 2,  # gCOD L-1
        "S2": 3,  # mmol L-1
        "Z": 4,  # mmol L-1
        "C": 5,  # mmol L-1 # Noted as mM in Hassam., check
        "pH": 6,  # Unused
    }
)

influent_state_units_dict = {
    "time": "Day",
    "D": "Day-1",
    "S1": "gCOD L-1",
    "S2": "mmol L-1",
    "Z": "mmol L-1",
    "C": "mmol L-1",
    "pH": "",
}
influent_state_units = [
    influent_state_units_dict[name] for name in influent_state_col.keys()
]


initial_state_col = sort_dict(
    {
        "X1": 0,  # gVS L-1 # conc. acidogenic bacteria
        "X2": 1,  # gVS L-1 # conc. methanogenic bacteria
        "S1": 2,  # gCOD L-1 # conc. substrate
        "S2": 3,  # mmol L-1 # conc. VFA
        "Z": 4,  # mmol L-1 # tot. alkalinity
        "C": 5,  # mmol L-1 # tot. inorg carbon conc. # Noted as mM in Hassam, check
    }
)

predict_col = sort_dict(
    {
        "time": 0,  # Day
        "X1": 1,  # gVS L-1 # conc. acidogenic bacteria
        "X2": 2,  # gVS L-1 # conc. methanogenic bacteria
        "S1": 3,  # gCOD L-1 # conc. substrate
        "S2": 4,  # mmol L-1 # conc. VFA
        "Z": 5,  # mmol L-1 # tot. alkalinity
        "C": 6,  # mmol L-1 # tot. inorg carbon conc.
        "qm": 7,  # mmol L-1 Day-1 # Noted as mM in Hassam, check
        "qc": 8,  # mmol L-1 Day-1 # carbon dioxide flow
    }
)

predict_units_dict = {
    "time": "Day",
    "X1": "gVS L-1",  # conc. acidogenic bacteria
    "X2": "gVS L-1",  # conc. methanogenic bacteria
    "S1": "gCOD L-1",  # conc. substrate
    "S2": "mmol L-1",  # conc. VFA
    "Z": "mmol L-1",  # tot. alkalinity
    "C": "mmol L-1",  # tot. inorg carbon conc.
    "qm": "mmol L-1 Day-1",  # Noted as mM in Hassam, check
    "qc": "mmol L-1 Day-1",  # carbon dioxide flow
}

predict_units = [predict_units_dict[name] for name in predict_col.keys()]

err_pred_col = ["S1", "S2", "qm", "qc"]
err_pred_ind = [predict_col[name] for name in err_pred_col]

parameter_dict = sort_dict(
    {
        "mu1max": 0,  # day-1 # max acidogenic growth rate
        "mu2max": 1,  # day-1 # max methanogenic growth rate
        "KS1": 2,  # gCOD L-1 # max saturation constant
        "KS2": 3,  # mmol L-1 # max saturation constant
        "KI2": 4,  # mmol L-1 # inhibition constant
    }
)

parameter_units = {
    "mu1max": "Day-1",
    "mu2max": "Day-1",
    "KS1": "gCOD L-1",
    "KS2": "mmol L-1",
    "KI2": "mmol L-1",
}


def param_names_to_index(param_names: list[str]) -> list[int]:
    """Transform a list of parameter names into the indexes of those parameters"""
    return [parameter_dict[param_name] for param_name in param_names]


def pred_names_to_index(pred_col_names: list[str]) -> list[int]:
    """Transform a list of prediction names into the indexes of those predictions"""
    return [predict_col[name] for name in pred_col_names]
