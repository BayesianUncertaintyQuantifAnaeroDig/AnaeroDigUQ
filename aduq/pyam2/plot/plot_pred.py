import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..IO._helper import predict_col, predict_units


def plot_pred(*preds: Tuple[np.ndarray], fold_path=".", aml_run=None) -> None:
    for i in range(len(predict_col) - 1):

        pred_name = list(predict_col.keys())[i + 1]
        for pred in preds:
            plt.plot(pred[:, 0], pred[:, i + 1])

        plt.ylabel(pred_name)
        plt.xlabel(f"Time ({predict_units[0]})")

        if predict_units[i + 1] != "":
            unit_text = f" ({predict_units[i+1]})"
        else:
            unit_text = ""
        plt.ylabel(f"{pred_name}" + unit_text)

        if aml_run is not None:
            aml_run.log_image(pred_name, plot=plt)
        plt.savefig(os.path.join(fold_path, f"{pred_name}.png"))
        plt.clf()
