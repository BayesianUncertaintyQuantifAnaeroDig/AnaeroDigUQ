"""
Class for Digester Information.

The necessary information about the digester configuration for the ADM1 routine to work is the
liquid phase volume, gas phase volume and Temperature.

The digester information can be loaded from a json file using load_dig_info.
The digester information can be saved to a json file using the .save method.
"""

import json
from typing import Optional


class DigesterInformation:
    """
    Class for Digester Information.

    Attributes:
        V_liq, the volume of the liquid phase in M3
        V_gas, the volume of the gas phase in M3
        T_ad, the volume inside the digester in Kelvin
    """

    def __init__(
        self, V_liq: float, V_gas: float, T_ad: float, T_op: Optional[float] = None
    ):
        assert V_liq > 0, "The liquid phase volume must be strictly positive"
        assert V_gas > 0, "The gas phase volume must be strictly positive"
        assert (
            T_ad > 0
        ), "The temperature of the digester must be strictly positive (in Kelvin)"

        self.V_liq = float(V_liq)
        self.V_gas = float(V_gas)
        self.T_ad = float(T_ad)
        if T_op is None:
            T_op = T_ad
        self.T_op = float(T_op)

    def save(self, path):
        """Save DigesterInformation object to .json file"""
        with open(path, "w") as file:
            json.dump(
                {
                    "V_liq": self.V_liq,
                    "V_gas": self.V_gas,
                    "T_ad": self.T_ad,
                    "T_op": self.T_op,
                },
                file,
            )

    def __str__(self):
        return str.join(
            "\n",
            [
                f"V_liq: {self.V_liq}",
                f"V_gas: {self.V_gas}",
                f"T_ad: {self.T_ad}",
                f"T_op: {self.T_op}",
            ],
        )

    def __repr__(self):
        return str.join(
            "\n",
            [
                f"V_liq: {self.V_liq}",
                f"V_gas: {self.V_gas}",
                f"T_ad: {self.T_ad}",
                f"T_op: {self.T_op}",
            ],
        )


def load_dig_info(path) -> DigesterInformation:
    with open(path, "r") as file:
        dig_info = json.load(file)

    if not "T_op" in dig_info.keys():
        dig_info["T_op"] = dig_info["T_ad"]

    return DigesterInformation(
        V_liq=dig_info["V_liq"],
        V_gas=dig_info["V_gas"],
        T_ad=dig_info["T_ad"],
        T_op=dig_info["T_op"],
    )
