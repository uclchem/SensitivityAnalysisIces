import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import uclchem


def run_UCLCHEM(
    outputfile_prefix: str,
    run_nr: str | int,
    temp: float,
    dens: float,
    zeta: float,
    radfield: float,
    force: bool,
) -> None:
    output_path = f"{outputfile_prefix}.h5"
    finalTime = 1e6
    if not force and os.path.isfile(output_path):
        with h5py.File(output_path, "r") as file:
            if run_nr in file:
                col_idx = [
                    col.decode("utf-8") for col in file["abundances_columns"]
                ].index("Time")
                maxTime = file[run_nr][-1, col_idx]
                if maxTime == finalTime:
                    print(
                        f"Detected that {output_path} run {run_nr} was already run, and reached maxTime {maxTime}"
                    )
                    return

    param_dict = {
        "endAtFinalDensity": False,  # stop at finalTime
        "freefall": False,  # don't increase density in freefall
        "initialDens": dens,  # starting density
        "initialTemp": temp,  # temperature of gas
        "finalTime": finalTime,  # final time
        "zeta": zeta,  # Cosmic ray ionisation rate as multiple of 1.3e-17 s^{-1}
        "radfield": radfield,
        "useCustomDiffusionBarriers": True,
        "useMinissaleIceChemdesEfficiency": False,
        "useTSTprefactors": False,  # required for nominal network to use HH prefactors, since we vary around that.
        # if we do a nominal run, use non-custom prefactors
        "useCustomPrefactors": run_nr != "nominal",
        "reltol": 1e-6,
        "abstol_factor": 1e-12,
    }
    if run_nr == "nominal":
        param_dict["rateFile"] = f"{outputfile_prefix}_rates.csv"

    modelStart = time.time()
    physicalDF, chemicalDF, _, result = uclchem.model.cloud(
        param_dict=param_dict, return_dataframe=True
    )
    modelEnd = time.time()

    if result < 0:
        print(
            f"Error in run {run_nr} at {temp} K, {dens} cm-3, {zeta * 1.3 * 1e-17} s-1, {radfield} Habing:"
        )
        print("Still writing its abundances to disk. Error:")
        error = uclchem.utils.check_error(result)
        print(f"\t{error}")

    df = pd.concat([physicalDF, chemicalDF], axis=1)

    if not os.path.isfile(output_path):
        with h5py.File(output_path, "w") as file:
            file.create_dataset(
                "abundances_columns",
                data=list(df.columns),
                compression="gzip",
                compression_opts=9,
            )

    float_cols = df.select_dtypes(np.float64).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    with h5py.File(output_path, "r+") as file:
        if run_nr in file.keys():
            del file[run_nr]
        file.create_dataset(
            run_nr, data=df.values, compression="gzip", compression_opts=9
        )

    if run_nr == "nominal":
        # Store the reaction rates in the h5 file as well.
        rates = pd.read_csv(f"{outputfile_prefix}_rates.csv")
        float_cols = rates.select_dtypes(np.float64).columns
        rates[float_cols] = rates[float_cols].astype(np.float32)
        with h5py.File(output_path, "a") as file:
            if "rates_columns" in file:
                del file["rates_columns"]
            file.create_dataset(
                "rates_columns",
                data=[col.strip() for col in rates.columns],
                compression="gzip",
                compression_opts=9,
            )
            if "nominal_rates" in file:
                del file["nominal_rates"]
            file.create_dataset(
                "nominal_rates",
                data=rates.values,
                compression="gzip",
                compression_opts=9,
            )
        # Remove the rates file
        os.remove(f"{outputfile_prefix}_rates.csv")

    conservation = uclchem.analysis.check_element_conservation(
        df, element_list=["H", "N", "C", "O", "S"]
    )
    print(
        f"Run {run_nr} at {temp} K, {dens} cm-3, {zeta * 1.3 * 1e-17:.1e} s-1, {radfield} Habing finished."
    )

    modelDuration = round(modelEnd - modelStart, 1)
    print(f"\tModel took {modelDuration} seconds")

    print(f"\tPercentage change in total abundances:\n\t\t{conservation}")


if __name__ == "__main__":
    output_prefix = sys.argv[1]
    run_nr = sys.argv[2]
    temp = float(sys.argv[3])
    dens = float(sys.argv[4])
    zeta = float(sys.argv[5])
    radfield = float(sys.argv[6])
    force = bool(int(sys.argv[7]))
    run_UCLCHEM(output_prefix, run_nr, temp, dens, zeta, radfield, force)
