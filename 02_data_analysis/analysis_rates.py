import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from uclchem.analysis import (
    analysis,
    analysis_solid_phase,
    get_change_df,
    getNetChange,
    read_output_file,
    read_rate_file,
)

from analysisTools import *

plt.style.use(["science", "style_AA", "color_bright"])

# Read parameter DataFrame. Contains information on what the values of the chemical parameters are for each sample
runsDir = "/data2/dijkhuis/ChemSamplingMC/setWidthProduction"
parametersDF = pd.read_csv(os.path.join(runsDir, "MC_parameter_runs.csv"), index_col=0)
parameterNames = parametersDF.columns

colors = getColors()

if True:
    spec = "CH3"
    on_grain = True

    nominalDF = read_output_file("TEST_sampleNominal_10.0_100000.0_1.0_1.0.csv")

    if on_grain:
        specAbunds = nominalDF["#" + spec]  #  + nominalDF["@" + spec]
    else:
        specAbunds = nominalDF[spec]
    rateDF = read_rate_file("TEST_sampleNominal_10.0_100000.0_1.0_1.0_rates.csv")
    changeDF = get_change_df(rateDF, spec, on_grain=on_grain)

    netChangeRateFile = getNetChange(changeDF) * 60.0 * 60.0 * 24.0 * 365.25

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(nominalDF["Time"], specAbunds)
    # axs[1].plot(FD_time, FD_slope)
    # axs[1].plot(changeDF_gas["Time"], netChangeRateFile)
    # axs[1].plot(nominalDF["Time"], netChange)

    reacs = [reac for reac in changeDF.columns if " LH " in reac and not "@" in reac]

    for i, reac in enumerate(reacs):
        axs[1].plot(
            changeDF["Time"],
            changeDF[reac],
            label=convertParameterNameToLegendLabel(reac),
            ls=["solid", "dashed", "-.", "dotted"][(i // len(colors)) % 4],
        )

    axs[1].legend()
    axs[0].set_xscale("log")
    axs[0].set_xlim([1e0, 1e6])
    axs[0].set_yscale("log")
    axs[1].set_yscale("symlog", linthresh=1e-35)
    plt.show()

    sys.exit()

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(nominalDF["Time"], specAbunds)
    reacCols = [col for col in ratesDataFrame.columns if "->" in col]
    alpha = 0.5
    for i, col in enumerate(reacCols):
        if not "FREEZE" in col:
            continue
        if col not in changeDF_gas.columns:
            print(f"AAAAAAAAAAAAAAAA {col}")
        print(ratesDataFrame[col])
        axs[1].plot(
            ratesDataFrame["Time"],
            np.abs(ratesDataFrame[col]),
            ls="solid",
            c=colors[i % len(colors)],
            alpha=alpha,
        )
        axs[1].plot(
            changeDF_gas["Time"],
            np.abs(changeDF_gas[col]),
            ls="dashed",
            c=colors[i % len(colors)],
            alpha=alpha,
        )
    axs[0].set_xscale("log")
    axs[0].set_xlim([1e0, 1e6])
    axs[0].set_yscale("log")
    axs[1].set_yscale("log")
    plt.show()

if False:
    spec = "C"
    on_grain = True

    file = "TEST_sampleNominal_50.0_1000.0_1.0_1.0.csv"

    nominalDF = read_output_file(file)

    rateDF = read_rate_file(file[:-4] + "_rates.csv")
    changeDF = get_change_df(rateDF, spec, on_grain=on_grain)

    reacs = ["#C + #O + LH -> #CO", "#SO2 + #C + LH -> #SO + #CO"]

    plt.figure()
    for i, reac in enumerate(reacs):
        splitReac = reac.split()
        lhDesEquivalent = reac.replace("#CO", "CO").replace("LH", "LHDES")
        if len(splitReac) == 9:
            reagent1Index = 6
            reagent2Index = 8

            lhdesReac1 = (
                splitReac[:reagent1Index]
                + [splitReac[reagent1Index].replace("#", "")]
                + splitReac[reagent1Index + 1 :]
            )
            lhdesReac1 = " ".join(lhdesReac1).replace("LH", "LHDES")

            lhdesReac2 = splitReac[:reagent2Index] + [
                splitReac[reagent2Index].replace("#", "")
            ]
            lhdesReac2 = " ".join(lhdesReac2).replace("LH", "LHDES")

            totalLH = changeDF[reac] + changeDF[lhdesReac1] + changeDF[lhdesReac2]
        else:
            totalLH = changeDF[reac] + changeDF[lhDesEquivalent]

        ratio = changeDF[lhDesEquivalent] / totalLH
        # print(ratio.values)

        eta_grain = ratio.iloc[2]
        fraction = -np.log(eta_grain)
        energyReleased = 1.0 / (
            fraction * ((120 - 28) / (120 + 28)) ** 2 / (1300.0 * 6)
        )
        # print(energyReleased)

        plt.plot(
            changeDF["Time"],
            ratio.values,
            label=convertParameterNameToLegendLabel(reac),
            c=colors[i],
            ls="solid",
        )
    plt.legend()
    plt.xlim([1e0, 1e6])
    plt.xscale("log")
    plt.show()

    plt.figure()
    for i, reac in enumerate(reacs):
        plt.plot(
            changeDF["Time"],
            changeDF[reac] * 60.0 * 60.0 * 24.0 * 365.25,
            label=convertParameterNameToLegendLabel(reac),
            c=colors[i],
            ls="solid",
        )

    spec = "CO"
    on_grain = False

    changeDF_gas = get_change_df(rateDF, spec, on_grain=on_grain)

    reacs = ["#C + #O + LHDES -> CO", "#SO2 + #C + LHDES -> #SO + CO"]
    for i, reac in enumerate(reacs):
        plt.plot(
            changeDF_gas["Time"],
            changeDF_gas[reac] * 60.0 * 60.0 * 24.0 * 365.25,
            # label=convertParameterNameToLegendLabel(reac),
            c=colors[i],
            ls="dashed",
        )
    plt.legend()
    plt.xlim([1e0, 1e6])
    plt.xscale("log")
    plt.yscale("log")
    plt.ylim([1e-20, 1e-8])

    plt.xlabel("Time (years)")
    plt.ylabel("Reaction rate (year$^{-1}$)")
    plt.tight_layout()
    plt.savefig("CO_correlation_SO2.pdf")
    plt.show()

if False:
    spec = "CO"
    on_grain = True

    file = "TEST_sampleNominal_50.0_1000.0_1.0_1.0.csv"

    nominalDF = read_output_file(file)

    rateDF = read_rate_file(file[:-4] + "_rates.csv")
    changeDF = get_change_df(rateDF, spec, on_grain=on_grain)

    reacs = [reac for reac in changeDF.columns if "LH ->" in reac and "#" in reac]

    plt.figure()
    for i, reac in enumerate(reacs):
        plt.plot(
            changeDF["Time"],
            changeDF[reac],
            label=convertParameterNameToLegendLabel(reac),
            ls="solid",
        )
    plt.legend()
    plt.xlim([1e0, 1e6])
    plt.xscale("log")
    plt.show()

if False:
    spec = "CO"
    on_grain = True

    file = "TEST_sampleNominal_50.0_1000000.0_1.0_1.0.csv"

    nominalDF = read_output_file(file)

    rateDF = read_rate_file(file[:-4] + "_rates.csv")
    changeDF_gas = get_change_df(rateDF, spec, on_grain=on_grain)

    reacs = [
        reac
        for reac in changeDF_gas.columns
        if "->" in reac
        and not "LHDES" in reac
        and not "@" in reac
        and not "CRP" in reac
        and not "DESOH" in reac
    ]

    specs = ["CO", "SO2", "HOCO", "HCO", "C", "O"]
    fig, axs = plt.subplots(2, 1, sharex=True)
    for i, spec in enumerate(specs):
        axs[0].plot(
            nominalDF["Time"],
            nominalDF["#" + spec],  #  + nominalDF["@" + spec],
            label=spec,
            c=colors[i],
        )
        axs[0].plot(
            nominalDF["Time"],
            nominalDF[spec],
            label=spec,
            c=colors[i],
            ls="dashed",
        )
    axs[0].legend()
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].set_xlim([1e0, 1e6])
    axs[0].set_ylim([1e-16, 1e-2])

    for i, reac in enumerate(reacs):
        is_formation = np.sign(changeDF_gas[reac].iloc[-1]) == 1
        if is_formation:
            ls = "solid"
        else:
            ls = "dashed"
        axs[1].plot(
            changeDF_gas["Time"],
            np.abs(changeDF_gas[reac]) * 60.0 * 60.0 * 24.0 * 365.25,
            label=reac.replace("#", ""),
            # c=colors[i],
            ls=ls,
        )

    axs[1].legend()
    axs[1].set_yscale("log")

    axs[1].set_xlabel("Time (years)")
    axs[1].set_ylabel("Reaction rate (year$^{-1}$)")
    plt.tight_layout()
    plt.savefig("CO_correlation_SO2.pdf")
    plt.show()

if False:
    spec = "O"
    on_grain = True

    rateDF = read_rate_file("TEST_sampleNominal_10.0_1000000.0_1.0_1.0_rates.csv")
    changeDF_gas = get_change_df(rateDF, spec, on_grain=on_grain)

    netChangeRateFile = getNetChange(changeDF_gas)

    reacs = [
        reac
        for reac in changeDF_gas.columns
        if "LH" in reac and not "LHDES" in reac and "#" in reac
    ]

    reacs = ["#N + #O + LH -> #NO", "#H + #O + LH -> #OH"]
    reacsDES = ["#N + #O + LHDES -> NO", "#H + #O + LHDES -> OH"]

    nominalDF = read_output_file("TEST_sampleNominal_10.0_1000000.0_1.0_1.0.csv")

    if on_grain:
        specAbunds = nominalDF["#" + spec] + nominalDF["@" + spec]
    else:
        specAbunds = nominalDF[spec]

    plt.figure()
    plt.plot(nominalDF["Time"], specAbunds)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([1e0, 1e6])
    plt.show()

    plt.figure()
    for i, reac in enumerate(reacs):
        plt.plot(
            changeDF_gas["Time"],
            -changeDF_gas[reac] * 60.0 * 60.0 * 24.0 * 365.25,
            label=convertParameterNameToLegendLabel(reac),
            c=colors[i],
            ls="solid",
        )

    O_freeze = "O + FREEZE -> #O"
    plt.plot(
        changeDF_gas["Time"],
        changeDF_gas[O_freeze] * 60.0 * 60.0 * 24.0 * 365.25,
        label="O Freeze-out",
        ls="solid",
        c=colors[2],
    )

    for i, reac in enumerate(reacsDES):
        plt.plot(
            changeDF_gas["Time"],
            -changeDF_gas[reac] * 60.0 * 60.0 * 24.0 * 365.25,
            c=colors[i],
            ls="dashed",
        )
    # This is like 80 orders of magnitude lower rates
    # reacsBulk = ["@N + @O + LH -> @NO", "@H + @O + LH -> @OH"]

    # for i, reac in enumerate(reacsBulk):
    #     plt.plot(
    #         changeDF["Time"],
    #         -changeDF[reac] * 60.0 * 60.0 * 24.0 * 365.25,
    #         label=convertParameterNameToLegendLabel(reac),
    #         # c=colors[i],
    #         ls="solid",
    #     )
    plt.legend()
    plt.xlim([1e0, 1e6])
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Time (years)")
    plt.ylabel("Reaction rate (year$^{-1}$)")
    plt.tight_layout()
    plt.savefig("H2O_correlation_N.pdf")
    plt.show()

if False:
    spec = "NH3"
    on_grain = True

    rateDF = read_rate_file("TEST_sampleNominal_10.0_1000000.0_1.0_1.0_rates.csv")
    changeDF_gas = get_change_df(rateDF, spec, on_grain=on_grain)

    netChangeRateFile = getNetChange(changeDF_gas)

    reacs = [
        reac
        for reac in changeDF_gas.columns
        if "LH" in reac and not "LHDES" in reac and "#" in reac
    ]

    nominalDF = read_output_file("TEST_sampleNominal_10.0_1000000.0_1.0_1.0.csv")

    if on_grain:
        specAbunds = nominalDF["#" + spec] + nominalDF["@" + spec]
    else:
        specAbunds = nominalDF[spec]

    plt.figure()
    plt.plot(nominalDF["Time"], specAbunds)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([1e0, 1e6])
    plt.show()

    plt.figure()
    for i, reac in enumerate(reacs):
        plt.plot(
            changeDF_gas["Time"],
            np.abs(changeDF_gas[reac] * 60.0 * 60.0 * 24.0 * 365.25),
            label=convertParameterNameToLegendLabel(reac),
            # c=colors[i],
            ls="solid",
        )

    plt.legend()
    plt.xlim([1e0, 1e6])
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Time (years)")
    plt.ylabel("Reaction rate (year$^{-1}$)")
    plt.tight_layout()
    # plt.savefig("H2O_correlation_N.pdf")
    plt.show()

if False:
    spec = "CH3OH"
    on_grain = True

    file = "TEST_sampleNominal_10.0_100000.0_1.0_1.0.csv"
    rateDF = read_rate_file(file[:-4] + "_rates.csv")
    changeDF_gas = get_change_df(rateDF, spec, on_grain=on_grain)

    reacs = [
        reac
        for reac in changeDF_gas.columns
        if "LH" in reac and not "LHDES" in reac and "#" in reac
    ]

    plt.figure()
    for i, reac in enumerate(reacs):
        plt.plot(
            changeDF_gas["Time"],
            np.abs(changeDF_gas[reac]) * 60.0 * 60.0 * 24.0 * 365.25,
            label=convertParameterNameToLegendLabel(reac),
            # c=colors[i],
            ls="solid",
        )

    plt.legend()
    plt.xlim([1e0, 1e6])
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Time (years)")
    plt.ylabel("Reaction rate (year$^{-1}$)")
    plt.tight_layout()
    plt.show()

if False:
    spec = "N"
    on_grain = True

    file = "TEST_sampleNominal_10.0_100000.0_1.0_1.0.csv"
    rateDF = read_rate_file(file[:-4] + "_rates.csv")
    changeDF_gas = get_change_df(rateDF, spec, on_grain=on_grain)

    reacs = [
        reac
        for reac in changeDF_gas.columns
        if "LH" in reac and not "LHDES" in reac and "#" in reac
    ]
    reacsDES = [
        reac for reac in changeDF_gas.columns if "LHDES" in reac and "#" in reac
    ]

    plt.figure()
    for i, reac in enumerate(reacs):
        plt.plot(
            changeDF_gas["Time"],
            np.abs(changeDF_gas[reac]) * 60.0 * 60.0 * 24.0 * 365.25,
            label=convertParameterNameToLegendLabel(reac),
            # c=colors[i],
            ls="solid",
        )
        plt.plot(
            changeDF_gas["Time"],
            np.abs(changeDF_gas[reacsDES[i]]) * 60.0 * 60.0 * 24.0 * 365.25,
            label=convertParameterNameToLegendLabel(reacsDES[i]),
            # c=colors[i],
            ls="solid",
        )

    plt.legend()
    plt.xlim([1e0, 1e6])
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Time (years)")
    plt.ylabel("Reaction rate (year$^{-1}$)")
    plt.tight_layout()
    plt.show()


if True:
    spec = "OH"
    on_grain = True

    file = "TEST_sampleNominal_30.0_1000.0_1.0_1.0.csv"
    rateDF = read_rate_file(file[:-4] + "_rates.csv")
    changeDF_gas = get_change_df(rateDF, spec, on_grain=on_grain)

    reacs = [
        reac
        for reac in changeDF_gas.columns
        if "LH" in reac and not "LHDES" in reac and "#" in reac
    ]

    plt.figure()
    for i, reac in enumerate(reacs):
        plt.plot(
            changeDF_gas["Time"],
            np.abs(changeDF_gas[reac]) * 60.0 * 60.0 * 24.0 * 365.25,
            label=convertParameterNameToLegendLabel(reac),
            # c=colors[i],
            ls="solid",
        )

    plt.legend()
    plt.xlim([1e0, 1e6])
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Time (years)")
    plt.ylabel("Reaction rate (year$^{-1}$)")
    plt.tight_layout()
    plt.show()
