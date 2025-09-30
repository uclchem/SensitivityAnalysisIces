import sys
import time

import pandas as pd
import uclchem


def run_UCLCHEM(
    outputfile_prefix: str,
    run_nr: str | int,
    temp: float,
    dens: float,
    zeta: float,
    radfield: float,
) -> None:
    param_dict = {
        "endAtFinalDensity": False,  # stop at finalTime
        "freefall": False,  # don't increase density in freefall
        "initialDens": dens,  # starting density
        "initialTemp": temp,  # temperature of gas
        "finalTime": 1.0e6,  # final time
        "zeta": zeta,  # Cosmic ray ionisation rate as multiple of 1.3e-17 s^{-1}
        "radfield": radfield,
        "useCustomDiffusionBarriers": True,
        "useMinissaleIceChemdesEfficiency": False,
        "useTSTprefactors": False,  # required for nominal network to use HH prefactors, since we vary around that.
        # if we do a nominal run, use non-custom prefactors
        "useCustomPrefactors": run_nr != "Nominal",
        "reltol": 1e-6,
        "abstol_factor": 1e-12,
    }

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

    outputfile = outputfile_prefix + f"_sample{run_nr}.h5"

    df = pd.concat([physicalDF, chemicalDF], axis=1)
    df.to_hdf(outputfile, key="abunds", format="fixed")

    conservation = uclchem.analysis.check_element_conservation(
        df, element_list=["H", "N", "C", "O", "S"]
    )
    print(
        f"Run {run_nr} at {temp} K, {dens} cm-3, {zeta * 1.3 * 1e-17} s-1, {radfield} Habing finished."
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
    run_UCLCHEM(output_prefix, run_nr, temp, dens, zeta, radfield)
