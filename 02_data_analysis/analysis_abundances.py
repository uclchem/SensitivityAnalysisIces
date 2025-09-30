import itertools
import os
import sys
import time
from copy import deepcopy

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from getEnergyReleasedFromReactions import *
from matplotlib.colors import LinearSegmentedColormap

# from inline_labels import add_inline_labels
from matplotlib.lines import Line2D
from uclchem.utils import get_reaction_table, get_species_table

from analysisTools import (
    calculateAllCorrelations2D,
    calculateHHprefactor,
    calculateRateConstantAllRows,
    calculateSignificantCorrelations2D,
    checkTimeIndices,
    compare_with_converged,
    convertDensityToTitle,
    convertParameterNameToAxisLabel,
    convertParameterNameToLegendLabel,
    convertSpeciesToLegendLabel,
    createRidgePlot,
    getAbundance,
    getAbundances,
    getAllRunsFilepaths,
    getColors,
    getConfidenceIntervalsOfAbundances,
    getDataFramesForPhysicalParamSet,
    getLogMeanOfAbundances,
    getPhysicalParamSets,
    getTimeFromIndices,
    getTimeIndices,
    getTotalRuntime,
    noCompetitionRateConstant,
    physicalParamSetToIndex,
    physicalParamSetToSaveString,
    physicalParamSetToString,
    plotDataFrameAbundances,
    rankInverseNormalTransform,
    removeModelsThatDontReachDesiredTime,
    test_convergence,
)

plt.style.use(["science", "color_muted", "font_cmubright"])

# Directory with all calculations
runsDir = "/data2/dijkhuis/ChemSamplingMC/setWidthProductionTight"

# Get filepaths of all model runs
filepathsSamples, filepathsNominal = getAllRunsFilepaths(runsDir, extension="h5")

# Get all sets of physical conditions
physicalParameterSets = getPhysicalParamSets(
    [filepaths[0] for filepaths in filepathsSamples]
)

# Calculate total runtime of models
totalRuntime = getTotalRuntime(list(itertools.chain(*filepathsSamples)))
totalRuntime /= 60.0 * 60.0 * 24.0
print(
    f"Total run time for all sample models:\n\t{round(totalRuntime, 2)} days, or\n\t{round(totalRuntime * 24, 2)} hours"
)

# Read parameter DataFrame. Contains information on what the values of the chemical parameters are for each sample
parametersDF = pd.read_csv(os.path.join(runsDir, "MC_parameter_runs.csv"), index_col=0)
parameterNames = parametersDF.columns

print(parametersDF["#H desprefac"])

speciesDF = get_species_table()
reactionsDF = get_reaction_table()

# Calculate rank-based inverse normal transform of every parameter
parameters = parametersDF.to_numpy()
parametersRIN = rankInverseNormalTransform(parameters)

colors = getColors()

# Colors from Paul Tol's bright colorscheme.
# NOMINAL_COLOR = "#FFAABB"  # light red, from medium contrast, but a bit brighter
# AVERAGE_COLOR = "#77AADD"  # light blue, from medium contrast, but a bit brighter
# NOMINAL_COLOR = "#F4A682"  # from BuRd
# AVERAGE_COLOR = "#6EA6CD"  # from Sunset
NOMINAL_COLOR = "#FDB366"  # from Sunset
AVERAGE_COLOR = "#60BCE9"  # from Nightfall
NOMINAL_LINESTYLE = "dashed"
AVERAGE_LINESTYLE = "solid"

ZERO_CORRELATION_ALPHA = 0.6
ZERO_CORRELATION_COLOR = "gray"
ZERO_CORRELATION_LS = "dashed"
ZERO_CORRELATION_LW = 1.0

WEAK_CORRELATION_ALPHA = 0.15
WEAK_CORRELATION_COLOR = "gray"

CORRELATION_CI_ALPHA = 0.25
CORRELATION_CI_EDGECOLOR = "face"
CORRELATION_CI_LINEWIDTH = 0.15

MARKER_EDGE_COLOR = "black"
MARKER_LINEWIDTH = 0.1

MIN_STATISTIC = 0.4

SAMPLING_95_CI = 0.08

N_JOBS = 10

PLOT_YLIM = [1e-16, 1e-2]
PLOT_XLIM = [1e0, 1e6]


paramTypes = ["diff", "diffprefac", "bind", "desprefac", "LH"]
paramTypesLegend = [
    r"$E_{\mathrm{diff}}$",
    r"$\nu_{\mathrm{diff}}$",
    r"$E_{\mathrm{bind}}$",
    r"$\nu_{\mathrm{des}}$",
    r"$E_{\mathrm{reac}}$",
]
paramTypesLinestyle = ["solid", "dashdot", "dashed", "dotted", (0, (5, 1))]


def createStandardPlot(
    sampleDFs: list[pd.DataFrame],
    nominalDF: pd.DataFrame,
    species: list[str],
    parametersRIN: np.ndarray,
    parameterNames: list[str],
    times: np.ndarray | list = np.logspace(0, 6, num=25),
    minStatistic: float = MIN_STATISTIC,
    on_grain: bool = False,
    savefigPath: str = None,
    quiet: bool = True,
    samplingCIError: float = 0.0,
    njobs: int = 1,
    doRINtransform: bool = True,
    parameters: None | np.ndarray = None,
    plot_individual_samples: bool = True,
):
    sampleDFs, correctIndices = removeModelsThatDontReachDesiredTime(
        sampleDFs, times[-1]
    )
    nSampleDFs = len(sampleDFs)

    # Because the dfs may not have the same timesteps in the files,
    # calculate at which index the times occur above for each file.
    timeIndices = getTimeIndices(sampleDFs, times)

    # Also get the times that those indices correspond to, such that we can plot them later.
    timeArray = getTimeFromIndices(sampleDFs, timeIndices)

    # Check timeIndices and timeArray, remove timeIndices that correspond to duplicate values of timeArray
    timeIndices, timeArray = checkTimeIndices(
        sampleDFs, timeArray, timeIndices, times, quiet=quiet
    )

    if not doRINtransform:
        if parameters is None:
            msg = "If doRINtransform is False, the parameters values must be given, not just the RIN transformed parameters"
            raise ValueError(msg)

        # find indices of prefactors
        prefactorIndices = [
            i
            for i, parameterName in enumerate(parameterNames)
            if "prefac" in parameterName
        ]

        parametersX = deepcopy(parameters)

        parametersX[:, prefactorIndices] = np.log10(parametersX[:, prefactorIndices])
    else:
        parametersX = parametersRIN

    fig, axs = plt.subplots(
        2,
        len(species),
        sharey="row",
        sharex="col",
        figsize=(4.5, 3),
        height_ratios=(1.0, 2.0 / 3.0),
    )
    seenParameters = []
    for i, spec in enumerate(species):
        axs[0, i].set_title(convertSpeciesToLegendLabel(spec), y=0.975)

        # Lightly plot all model runs
        if plot_individual_samples:
            plotDataFrameAbundances(
                sampleDFs,
                spec,
                on_grain=on_grain,
                ax=axs[0, i],
            )

        # Also plot nominal model (model with standard network)
        if on_grain:
            nominalAbundance = nominalDF["#" + spec] + nominalDF["@" + spec]
        else:
            nominalAbundance = nominalDF[spec]
        axs[0, i].plot(
            nominalDF["Time"],
            nominalAbundance,
            c=NOMINAL_COLOR,
            ls=NOMINAL_LINESTYLE,
        )

        # Get the abundances of current species at all timesteps in all model runs
        abundances = getAbundances(
            sampleDFs, timeIndices, spec=spec, on_grain=on_grain, njobs=njobs
        )

        # Plot log-spaced average
        logMeanAbundances = getLogMeanOfAbundances(abundances)
        axs[0, i].plot(
            timeArray[0, :],
            logMeanAbundances,
            c=AVERAGE_COLOR if plot_individual_samples else "k",
            ls=AVERAGE_LINESTYLE,
        )

        if not plot_individual_samples:
            # Plot confidence interval of abundances
            for confidence_level in [0.6827, 0.9545, 1.0]:
                cilow, cihigh = getConfidenceIntervalsOfAbundances(
                    abundances, confidence_level=confidence_level
                )
                axs[0, i].fill_between(
                    timeArray[0, :],
                    cilow,
                    cihigh,
                    color="k",
                    alpha=0.18,
                    edgecolor="none",
                )

        # Perform rank-based inverse normal transformation on the abundances
        if doRINtransform:
            abundancesRIN = rankInverseNormalTransform(abundances)
        else:
            abundancesRIN = np.log10(abundances)

        # Add horizontal line at correlation = 0, and fill areas where correlation < minStatistic
        axs[1, i].axhline(
            0,
            color=ZERO_CORRELATION_COLOR,
            ls=ZERO_CORRELATION_LS,
            lw=ZERO_CORRELATION_LW,
            alpha=ZERO_CORRELATION_ALPHA,
            zorder=0,
        )
        axs[1, i].fill_between(
            [0, 1e10],
            [-minStatistic] * 2,
            [minStatistic] * 2,
            color=WEAK_CORRELATION_COLOR,
            alpha=WEAK_CORRELATION_ALPHA,
            edgecolor="none",
            zorder=0,
        )

        sigCorrelations = calculateSignificantCorrelations2D(
            abundancesRIN,
            parametersX[correctIndices, :],
            parameterNames,
            confidence_level=0.95,
            minStatistic=minStatistic,
        )

        if sigCorrelations is None:
            # If there are no strong enough correlations, go to next species
            continue

        # For all significant and strong correlations, plot them.
        for j, row in sigCorrelations.iterrows():
            plot_kwargs = {}
            if row["parameter"] not in seenParameters:
                seenParameters.append(row["parameter"])
                plot_kwargs["label"] = convertParameterNameToLegendLabel(
                    row["parameter"]
                )
            colorIndex = (seenParameters.index(row["parameter"])) % len(colors)
            axs[1, i].plot(
                sampleDFs[0]["Time"].iloc[timeIndices[0, :]],
                row["statistic"],
                c=colors[colorIndex],
                **plot_kwargs,
            )

            axs[1, i].fill_between(
                sampleDFs[0]["Time"].iloc[timeIndices[0, :]],
                np.array(row["cilow"]) - samplingCIError,
                np.array(row["cihigh"]) + samplingCIError,
                alpha=CORRELATION_CI_ALPHA,
                color=colors[colorIndex],
                edgecolor=CORRELATION_CI_EDGECOLOR,
                linewidth=CORRELATION_CI_LINEWIDTH,
            )
        leg = axs[1, i].legend(
            handlelength=0,
            handletextpad=0,
            labelcolor="linecolor",
            loc="lower left",
        )
        [handle.set_visible(False) for handle in leg.legend_handles]

    for ax in axs.flat:
        ax.set_xlim([1e0, 1e6])
        ax.set_xscale("log")
        ax.label_outer()

    axs[0, 0].set_ylim([1e-14, 1e-2])
    axs[0, 0].set_yscale("log")
    axs[1, 0].set_ylim([-1.2, 1.2])
    axs[1, 0].set_yticks([-1, 0, 1])
    axs[0, 0].set_ylabel("Abundance (wrt H)")
    axs[1, 0].set_ylabel(r"$r_{\mathrm{RIN}}$")

    if len(species) == 5:
        fig.supxlabel("Time (years)", x=0.525)
    elif len(species) == 4:
        fig.supxlabel("Time (years)")
    else:
        fig.supxlabel("Time (years)")

    # Set xticks
    axs[0, 0].set_xticks([1e0, 1e2, 1e4, 1e6])
    [axs[0, i].set_xticks([1e2, 1e4, 1e6]) for i in range(1, len(species))]
    [
        axs[0, i].set_xticks([1e1, 1e3, 1e5], ["", "", ""], minor=True)
        for i in range(len(species))
    ]

    plt.subplots_adjust(hspace=0, wspace=0, bottom=0.117, left=0.15, top=0.98)
    if savefigPath is not None:
        if not "." in savefigPath:
            savefigPath += ".pdf"
        plt.savefig(savefigPath)
    else:
        plt.show()
    plt.close()


if False:
    spec = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    on_grain = True
    T = 10.0
    nH = 1e5
    UV = 1.0
    zeta = 1.0
    physicalParamIndex = physicalParamSetToIndex(
        physicalParameterSets, T=T, nH=nH, zeta=zeta, radfield=UV
    )
    physicalParams = physicalParameterSets[physicalParamIndex]
    print(f"Running analysis on {physicalParamSetToString(physicalParams)}")

    # Get filepaths of all model runs at this certain set of physical conditions
    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physicalParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
    )

    createStandardPlot(
        sampleDFs,
        nominalDF,
        spec,
        parametersRIN,
        parameterNames,
        on_grain=on_grain,
        njobs=N_JOBS,
        minStatistic=MIN_STATISTIC,
        samplingCIError=SAMPLING_95_CI,
    )

if False:
    specs = ["H2O", "CO", "CO2", "CH3OH"]
    on_grain = True
    T = 10.0
    nH = 1e4
    UV = 1.0
    zeta = 1.0
    physicalParamIndex = physicalParamSetToIndex(
        physicalParameterSets, T=T, nH=nH, zeta=zeta, radfield=UV
    )
    physicalParams = physicalParameterSets[physicalParamIndex]
    print(f"Running analysis on {physicalParamSetToString(physicalParams)}")

    # Get filepaths of all model runs at this certain set of physical conditions
    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physicalParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
    )

    timeIndices = getTimeIndices(sampleDFs, 1e6)
    abundances = np.empty(shape=(len(sampleDFs), len(specs)))
    for i, spec in enumerate(specs):
        abundances[:, i] = getAbundances(
            sampleDFs, timeIndices, spec, on_grain=on_grain
        )

    plt.figure()
    createViolinPlot(abundances, specs, color=colors[: len(specs)], connect_means=True)
    plt.show()

if False:
    specs = ["N2", "NH3", "CN", "C", "CH"]
    T = 10.0
    nH = 1e4
    UV = 1.0
    zeta = 1.0

    physicalParamIndex = physicalParamSetToIndex(
        physicalParameterSets, T=T, nH=nH, zeta=zeta, radfield=UV
    )
    physicalParams = physicalParameterSets[physicalParamIndex]
    print(f"Running analysis on {physicalParamSetToString(physicalParams)}")

    # Get filepaths of all model runs at this certain set of physical conditions
    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physicalParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
    )

    fig, axs = plt.subplots(1, 3, figsize=(4.5, 2.6), sharex=True, sharey=True)
    for i, spec in enumerate(specs):
        axs[0].plot(
            nominalDF["Time"],
            nominalDF[spec],
            c=colors[i],
            label=convertSpeciesToLegendLabel(spec),
        )
        axs[1].plot(
            nominalDF["Time"],
            nominalDF["#" + spec],
            c=colors[i],
            label=convertSpeciesToLegendLabel(spec),
        )
        axs[2].plot(
            nominalDF["Time"],
            nominalDF["@" + spec],
            c=colors[i],
            label=convertSpeciesToLegendLabel(spec),
        )
    axs[0].set_xscale("log")
    axs[0].set_xlim([1e0, 1e6])
    axs[0].set_yscale("log")
    [axs[i].set_xlabel("Time (years)") for i in range(3)]
    axs[0].set_title("Gas")
    axs[1].set_title("Surface")
    axs[2].set_title("Bulk")
    axs[0].set_ylabel("Abundance (wrt H nuclei)")
    axs[0].set_ylim([1e-14, 1e-2])
    axs[2].legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

    plt.savefig("nitrogen_bearing_species.pdf")
    plt.show()

if False:
    spec = "CO"
    on_grain = True
    T = 10.0
    nH = 1e5
    UV = 1.0
    zeta = 1.0
    physicalParamIndex = physicalParamSetToIndex(
        physicalParameterSets, T=T, nH=nH, zeta=zeta, radfield=UV
    )
    physicalParams = physicalParameterSets[physicalParamIndex]
    print(f"Running analysis on {physicalParamSetToString(physicalParams)}")

    # Get filepaths of all model runs at this certain set of physical conditions
    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physicalParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
    )

    times = np.logspace(3, 6, num=10, endpoint=True)
    timeIndices = getTimeIndices(sampleDFs, times)
    abundances = getAbundances(sampleDFs, timeIndices, spec, on_grain=on_grain)

    iridescent_colors = [
        "#FEFBE9",
        "#FCF7D5",
        "#F5F3C1",
        "#EAF0B5",
        "#DDECBF",
        "#D0E7CA",
        "#C2E3D2",
        "#B5DDD8",
        "#A8D8DC",
        "#9BD2E1",
        "#8DCBE4",
        "#81C4E7",
        "#7BBCE7",
        "#7EB2E4",
        "#88A5DD",
        "#9398D2",
        "#9B8AC4",
        "#9D7DB2",
        "#9A709E",
        "#906388",
        "#805770",
        "#684957",
        "#46353A",
    ]
    cmap = LinearSegmentedColormap.from_list("iridescent", iridescent_colors)

    plt.figure()
    createRidgePlot(
        abundances,
        overlap=0.5,
        marker_step=1,
        padding_x=0.5,
        fill=True,
        color=cmap,
        npoints=500,
    )
    plt.tight_layout()
    plt.show()


if True:
    # Species to plot and calculate
    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    on_grain = (
        True  # If this is false, the top plots will not plot correct abundances atm
    )

    # Physical condition set index
    T = 50.0
    nH = 1e5
    UV = 1.0
    zeta = 1.0
    physicalParamIndex = physicalParamSetToIndex(
        physicalParameterSets, T=T, nH=nH, zeta=zeta, radfield=UV
    )

    # Times when to calculate the correlations
    times = np.logspace(0, 6, num=100)

    physicalParams = physicalParameterSets[physicalParamIndex]
    print(f"Running analysis on {physicalParamSetToString(physicalParams)}")

    # Get filepaths of all model runs at this certain set of physical conditions
    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physicalParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
    )
    imageSaveString = (
        f"sensitivities_{physicalParamSetToSaveString(physicalParams)}.pdf"
    )
    createStandardPlot(
        sampleDFs,
        nominalDF,
        species,
        parametersRIN,
        parameterNames,
        samplingCIError=SAMPLING_95_CI,
        times=times,
        on_grain=on_grain,
        minStatistic=0.5,
        njobs=N_JOBS,
        savefigPath=imageSaveString,
        plot_individual_samples=False,
    )


def getReactionSeries(reactionDF, reaction):
    splitReaction = reaction.split()

    for i, row in reactionDF.iterrows():
        if row["Reactant 1"] != splitReaction[0]:
            continue
        if row["Reactant 2"] != splitReaction[2]:
            continue
        print(row, reaction)
        if row["Product 1"] != splitReaction[6]:
            continue
        if len(splitReaction) > 7 and row["Product 2"] != splitReaction[8]:
            continue
        return row


if False:
    temp = 10.0

    alpha = 0.75
    markersize = 7

    reactions = [
        "#H + #CH2OH + LH -> #CH3OH",
        "#H + #H2CO + LH -> #CH3O",
        "#H + #CH3OH + LH -> #CH2OH + #H2",
        "#H + #CH3OH + LH -> #CH3O + #H2",
    ]
    barriers = [0, 1900, 3891, 4878]

    param = "#H diff"

    colors = [
        "#004488",
        "#77AADD",
        "#994455",
        "#FFAABB",
    ]  # Colors from Paul Tol's Medium
    markers = ["P", "X", "v", "^"]

    values = parametersDF[param]
    min, max, average = np.min(values), np.max(values), np.average(values)

    hSeries = speciesDF.iloc[list(speciesDF["NAME"].values).index("#H")]
    hPrefac = calculateHHprefactor(hSeries)
    customHdiff = np.linspace(0, 600)
    hDiffusionRate = hPrefac * np.exp(-customHdiff / temp) * 60.0 * 60.0 * 24.0 * 365.25

    plt.figure()
    plt.plot(customHdiff, hDiffusionRate, c="gray", ls="dashed", zorder=-1, alpha=0.6)

    for i, reaction in enumerate(reactions):
        rateConstants = calculateRateConstantAllRows(
            parametersDF,
            reaction,
            temp,
            only_competition_fraction=False,
            tunnelingMass=1,
        )

        # reactionSeries = getReactionSeries(reactionsDF, reaction)
        # if reactionSeries is None:
        #     print(reactionsDF)
        #     print(reaction)
        #     raise ValueError()
        rateConstantNoComp = (
            noCompetitionRateConstant(hPrefac, barriers[i], 1.0, 10.0)
            * 60.0
            * 60.0
            * 24.0
            * 365.25
        )
        plt.axhline(
            rateConstantNoComp,
            c=colors[i],
            ls="solid",
            zorder=-1,
            alpha=alpha,
        )

        plt.scatter(
            parametersDF[param].values,
            rateConstants * 60.0 * 60.0 * 24.0 * 365.25,
            s=markersize,
            c=colors[i],
            edgecolor=MARKER_EDGE_COLOR,
            linewidths=MARKER_LINEWIDTH,
            label=convertParameterNameToLegendLabel(reaction),
            alpha=alpha,
            marker=markers[i],
        )

    my_labels = [convertParameterNameToLegendLabel(reaction) for reaction in reactions]
    plt.legend(
        loc="lower center",
        ncols=2,
        bbox_to_anchor=(0.45, 1.0),
        handlelength=0.5,
        handletextpad=0.25,
        columnspacing=1.0,
        labelcolor=colors[: len(my_labels)],
        markerscale=1.8,
    )
    plt.xlim([min - 20, max + 20])
    plt.xlabel(convertParameterNameToAxisLabel(param))
    plt.ylabel("Rate constant (unit year$^{-1}$)")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("rateConstantsCH3OH.pdf")
    plt.show()

if False:
    specs = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    on_grain = True
    use_RIN = True
    only_surf = False
    only_bulk = False

    # colorsTimes = ["#81C4E7", "#9398D2", "#906388"]
    colorsTimes = ["#7BBCE7", "#9B8AC4", "#805770"]
    times = [1e3, 1e4, 1e5]
    density = 1e5
    zeta = 1.0
    UV = 1.0
    temp = 10.0
    quiet = True

    alpha = 1.0
    markersize = 10

    param = "#H diff"

    paramValues = parametersDF[param].values
    paramAverage = np.average(paramValues)

    if use_RIN:
        paramValuesRIN = rankInverseNormalTransform(paramValues)
    else:
        paramValuesRIN = paramValues

    paramValuesRIN = paramValuesRIN.reshape((len(paramValuesRIN), 1))

    physicalParamIndex = physicalParamSetToIndex(
        physicalParameterSets,
        T=temp,
        nH=density,
        zeta=zeta,
        radfield=UV,
    )
    physicalParams = physicalParameterSets[physicalParamIndex]
    print(f"Running analysis on {physicalParamSetToString(physicalParams)}")

    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physicalParamIndex,
        filepathsSamples,
        filepathsNominal,
        njobs=N_JOBS,
    )

    sampleDFs, correctIndices = removeModelsThatDontReachDesiredTime(
        sampleDFs, times[-1]
    )

    # Because the dfs may not have the same timesteps in the files,
    # calculate at which index the times occur above for each file.
    timeIndices = getTimeIndices(sampleDFs, times)

    # Also get the times that those indices correspond to, such that we can plot them later.
    timeArray = getTimeFromIndices(sampleDFs, timeIndices)

    # Check timeIndices and timeArray, remove timeIndices that correspond to duplicate values of timeArray
    timeIndices, timeArray = checkTimeIndices(
        sampleDFs, timeArray, timeIndices, times, quiet=quiet
    )

    timeIndicesNominal = getTimeIndices([nominalDF], times)

    fig, axs = plt.subplots(1, len(specs), sharex=True, sharey=True, figsize=(6, 3.0))
    for i, spec in enumerate(specs):
        axs[i].set_title(convertSpeciesToLegendLabel(spec), y=0.985)

        abundancesSamples = getAbundances(
            sampleDFs,
            timeIndices,
            spec,
            on_grain=on_grain,
            njobs=N_JOBS,
            only_surf=only_surf,
            only_bulk=only_bulk,
        )

        abundancesNominal = getAbundances(
            [nominalDF],
            timeIndicesNominal,
            spec,
            on_grain=on_grain,
            njobs=N_JOBS,
            only_surf=only_surf,
            only_bulk=only_bulk,
        )[0, :]

        if use_RIN:
            abundancesRIN = rankInverseNormalTransform(abundancesSamples)
        else:
            abundancesRIN = np.log10(abundancesSamples)

        allCorrelations = calculateAllCorrelations2D(
            abundancesRIN,
            paramValuesRIN,
            [param],
            confidence_level=0.95,
            calculateConfidenceInterval=True,
        )

        for j, time in enumerate(times):
            hdiffValues = parametersDF["#H diff"].values
            # alphas = [
            #     0.0 if paramValues[i] > hdiffValues[i] else 1.0
            #     for i in range(len(paramValues))
            # ]
            axs[i].scatter(
                paramValues,
                abundancesSamples[:, j],
                c=colorsTimes[j],
                alpha=alpha,
                s=markersize,
                edgecolor="none",
                marker=".",
                # label=f"$r_{{\\mathrm{{RIN}}}}={{{round(allCorrelations['statistic'].iloc[0][j], 2)}}}\pm{{{round(allCorrelations['cidiffdown'].iloc[0][j], 2)}}}$",
                label=f"${{{round(allCorrelations['statistic'].iloc[0][j], 2)}}}\pm{{{round(allCorrelations['cidiffdown'].iloc[0][j] + SAMPLING_95_CI, 2)}}}$",
            )

            axs[i].scatter(
                paramAverage,
                abundancesNominal[j],
                c=colorsTimes[j],
                alpha=alpha,
                s=markersize,
                marker="X",
                edgecolor="k",
            )

        leg = axs[i].legend(
            handlelength=0,
            handletextpad=0,
            bbox_to_anchor=(0.975, 0.0),
            loc="lower right",
            alignment="right",
            labelspacing=0.25,
            fontsize="small",
            labelcolor=[colorsTimes[k] for k in range(3)],
        )
        [handle.set_visible(False) for handle in leg.legend_handles]

        max_shift = max([t.get_window_extent().width for t in leg.get_texts()])
        for t in leg.get_texts():
            t.set_ha("right")  # ha is alias for horizontalalignment
            temp_shift = max_shift - t.get_window_extent().width
            t.set_position((temp_shift, 0))

        # axs[i].set_title(convertSpeciesToLegendLabel(spec), y=0.8)

    if param == "#H diff":
        axs[i].set_xticks([250, 400])

    axs[0].set_ylim([1e-18, 1e-2])
    axs[0].set_yscale("log")
    axs[0].set_yticks([1e-16, 1e-12, 1e-8, 1e-4], [""] * 4, minor=True)
    axs[0].set_yticks([1e-18, 1e-14, 1e-10, 1e-6, 1e-2], minor=False)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.supxlabel(convertParameterNameToAxisLabel(param), x=0.516)
    fig.supylabel("Abundance (wrt H)")
    if use_RIN:
        plt.savefig("correlationsWithHdiff_RIN.pdf")
    else:
        plt.savefig("correlationsWithHdiff_noRIN.pdf")
    plt.show()


if False:
    specs = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    on_grain = True
    use_RIN = True
    only_surf = False
    only_bulk = False

    # colorsTimes = ["#81C4E7", "#9398D2", "#906388"]
    colorsTimes = ["#000000"]  # , "#9B8AC4", "#805770"]
    times = [1e3]  # , 1e4, 1e5]
    density = 1e5
    zeta = 1.0
    UV = 1.0
    temp = 10.0
    quiet = True

    alpha = 0.5
    markersize = 10

    param = "#H diff"

    paramValues = parametersDF[param].values
    paramAverage = np.average(paramValues)

    if use_RIN:
        paramValuesRIN = rankInverseNormalTransform(paramValues)
    else:
        paramValuesRIN = paramValues

    paramValuesRIN = paramValuesRIN.reshape((len(paramValuesRIN), 1))

    physicalParamIndex = physicalParamSetToIndex(
        physicalParameterSets,
        T=temp,
        nH=density,
        zeta=zeta,
        radfield=UV,
    )
    physicalParams = physicalParameterSets[physicalParamIndex]
    print(f"Running analysis on {physicalParamSetToString(physicalParams)}")

    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physicalParamIndex,
        filepathsSamples,
        filepathsNominal,
        njobs=N_JOBS,
    )

    sampleDFs, correctIndices = removeModelsThatDontReachDesiredTime(
        sampleDFs, times[-1]
    )

    # Because the dfs may not have the same timesteps in the files,
    # calculate at which index the times occur above for each file.
    timeIndices = getTimeIndices(sampleDFs, times)

    # Also get the times that those indices correspond to, such that we can plot them later.
    timeArray = getTimeFromIndices(sampleDFs, timeIndices)

    # Check timeIndices and timeArray, remove timeIndices that correspond to duplicate values of timeArray
    timeIndices, timeArray = checkTimeIndices(
        sampleDFs, timeArray, timeIndices, times, quiet=quiet
    )

    timeIndicesNominal = getTimeIndices([nominalDF], times)

    fig, axs = plt.subplots(1, len(specs), sharex=True, sharey=True, figsize=(6, 3.0))
    for i, spec in enumerate(specs):
        axs[i].set_title(convertSpeciesToLegendLabel(spec), y=0.985)

        abundancesSamples = getAbundances(
            sampleDFs,
            timeIndices,
            spec,
            on_grain=on_grain,
            njobs=N_JOBS,
            only_surf=only_surf,
            only_bulk=only_bulk,
        )

        abundancesNominal = getAbundances(
            [nominalDF],
            timeIndicesNominal,
            spec,
            on_grain=on_grain,
            njobs=N_JOBS,
            only_surf=only_surf,
            only_bulk=only_bulk,
        )[0, :]

        if use_RIN:
            abundancesRIN = rankInverseNormalTransform(abundancesSamples)
        else:
            abundancesRIN = np.log10(abundancesSamples)

        allCorrelations = calculateAllCorrelations2D(
            abundancesRIN,
            paramValuesRIN,
            [param],
            confidence_level=0.95,
            calculateConfidenceInterval=True,
        )

        for j, time in enumerate(times):
            hdiffValues = parametersDF["#H diff"].values
            # alphas = [
            #     0.0 if paramValues[i] > hdiffValues[i] else 1.0
            #     for i in range(len(paramValues))
            # ]
            axs[i].scatter(
                paramValues,
                abundancesSamples[:, j],
                c=colorsTimes[j],
                alpha=alpha,
                s=markersize,
                edgecolor="none",
                marker=".",
                # label=f"$r_{{\\mathrm{{RIN}}}}={{{round(allCorrelations['statistic'].iloc[0][j], 2)}}}\pm{{{round(allCorrelations['cidiffdown'].iloc[0][j], 2)}}}$",
                label=f"${{{round(allCorrelations['statistic'].iloc[0][j], 2)}}}\pm{{{round(allCorrelations['cidiffdown'].iloc[0][j] + SAMPLING_95_CI, 2)}}}$",
            )

            axs[i].scatter(
                paramAverage,
                abundancesNominal[j],
                c=NOMINAL_COLOR,
                alpha=1.0,
                s=markersize * 5,
                marker="X",
                edgecolor=NOMINAL_COLOR,
            )

        leg = axs[i].legend(
            handlelength=0,
            handletextpad=0,
            bbox_to_anchor=(0.5, 0.0),
            loc="lower center",
            alignment="center",
            labelspacing=0.25,
            fontsize="small",
            labelcolor=[colorsTimes[k] for k in range(len(times))],
        )
        [handle.set_visible(False) for handle in leg.legend_handles]

    if param == "#H diff":
        axs[i].set_xticks([250, 400])

    axs[0].set_ylim([1e-18, 1e-2])
    axs[0].set_yscale("log")
    axs[0].set_yticks([1e-16, 1e-12, 1e-8, 1e-4], [""] * 4, minor=True)
    axs[0].set_yticks([1e-18, 1e-14, 1e-10, 1e-6, 1e-2], minor=False)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.supxlabel(convertParameterNameToAxisLabel(param), x=0.516, y=-0.01)
    fig.supylabel("Abundance (wrt H)")
    if use_RIN:
        plt.savefig("correlationsWithHdiff_RIN.pdf")
    else:
        plt.savefig("correlationsWithHdiff_noRIN.pdf")
    plt.show()

if False:
    specs = ["H2O", "CO", "CO2", "CH3OH", "NH3", "CH4"]
    on_grain = True

    times = [1e4, 1e5, 1e6]
    density = 1e5
    zeta = 1.0
    UV = 1.0
    temp = 10.0

    alpha = 0.5
    markersize = 5

    param = "#N diff"
    paramValues = parametersDF[param].values

    param2 = "#H diff"
    paramValues2 = parametersDF[param2].values
    paramValues = paramValues - paramValues2

    paramValuesRIN = rankInverseNormalTransform(paramValues)
    paramValuesRIN = paramValuesRIN.reshape((len(paramValuesRIN), 1))

    quiet = True

    fig, axs = plt.subplots(2, 3, figsize=(4, 3), sharex=True, sharey=True)
    for i, spec in enumerate(specs):
        physicalParamIndex = physicalParamSetToIndex(
            physicalParameterSets,
            T=temp,
            nH=density,
            zeta=zeta,
            radfield=UV,
        )
        physicalParams = physicalParameterSets[physicalParamIndex]
        print(f"Running analysis on {physicalParamSetToString(physicalParams)}")

        sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
            physicalParamIndex,
            filepathsSamples,
            filepathsNominal,
            njobs=N_JOBS,
        )

        sampleDFs, correctIndices = removeModelsThatDontReachDesiredTime(
            sampleDFs, times[-1]
        )

        # Because the dfs may not have the same timesteps in the files,
        # calculate at which index the times occur above for each file.
        timeIndices = getTimeIndices(sampleDFs, times)

        # Also get the times that those indices correspond to, such that we can plot them later.
        timeArray = getTimeFromIndices(sampleDFs, timeIndices)

        # Check timeIndices and timeArray, remove timeIndices that correspond to duplicate values of timeArray
        timeIndices, timeArray = checkTimeIndices(
            sampleDFs, timeArray, timeIndices, times, quiet=quiet
        )

        abundancesSamples = getAbundances(
            sampleDFs, timeIndices, spec, on_grain=on_grain, njobs=N_JOBS
        )

        abundancesRIN = rankInverseNormalTransform(abundancesSamples)
        allCorrelations = calculateAllCorrelations2D(
            abundancesRIN,
            paramValuesRIN,
            [param],
            confidence_level=0.95,
            calculateConfidenceInterval=True,
        )

        for j, time in enumerate(times):
            axs[i // 3, i % 3].scatter(
                paramValues,
                abundancesSamples[:, j],
                c=colors[j],
                alpha=alpha,
                s=markersize,
                marker=".",
                # label=f"$r_{{\\mathrm{{RIN}}}}={{{round(allCorrelations['statistic'].iloc[0][j], 2)}}}\pm{{{round(allCorrelations['cidiffdown'].iloc[0][j], 2)}}}$",
                label=f"${{{round(allCorrelations['statistic'].iloc[0][j], 2)}}}\pm{{{round(allCorrelations['cidiffdown'].iloc[0][j], 2)}}}$",
            )

        leg = axs[i // 3, i % 3].legend(
            handlelength=0,
            handletextpad=0,
            bbox_to_anchor=(0.975, 0.025),
            loc="lower right",
            alignment="right",
            labelcolor=[colors[i] for i in range(3)],
        )
        [handle.set_visible(False) for handle in leg.legend_handles]
        [t.set_ha("right") for t in leg.get_texts()]

        axs[i // 3, i % 3].set_title(convertSpeciesToLegendLabel(spec), y=0.8)
    axs[0, 0].set_ylim([1e-16, 1e-2])
    axs[0, 0].set_yscale("log")
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.supxlabel(convertParameterNameToAxisLabel(param))
    fig.supylabel("Abundance (wrt H)")
    # plt.savefig("correltaionsWithHdiff.pdf")
    plt.show()


if False:
    spec = "NH3"
    on_grain = True

    times = [1e1, 1e2, 1e3]
    density = 1e4
    zeta = 1.0
    UV = 1.0
    temp = 10.0

    alpha = 0.5
    markersize = 5

    param = "#NH bind"
    paramValuesRIN = rankInverseNormalTransform(parametersDF[param].values)
    paramValuesRIN = paramValuesRIN.reshape((len(paramValuesRIN), 1))

    quiet = True

    plt.figure()

    physicalParamIndex = physicalParamSetToIndex(
        physicalParameterSets,
        T=temp,
        nH=density,
        zeta=zeta,
        radfield=UV,
    )
    physicalParams = physicalParameterSets[physicalParamIndex]
    print(f"Running analysis on {physicalParamSetToString(physicalParams)}")

    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physicalParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
    )

    sampleDFs, correctIndices = removeModelsThatDontReachDesiredTime(
        sampleDFs, times[-1]
    )

    # Because the dfs may not have the same timesteps in the files,
    # calculate at which index the times occur above for each file.
    timeIndices = getTimeIndices(sampleDFs, times)

    # Also get the times that those indices correspond to, such that we can plot them later.
    timeArray = getTimeFromIndices(sampleDFs, timeIndices)

    # Check timeIndices and timeArray, remove timeIndices that correspond to duplicate values of timeArray
    timeIndices, timeArray = checkTimeIndices(
        sampleDFs, timeArray, timeIndices, times, quiet=quiet
    )

    abundancesSamples = getAbundances(
        sampleDFs, timeIndices, spec, on_grain=on_grain, njobs=N_JOBS
    )

    abundancesRIN = rankInverseNormalTransform(abundancesSamples)
    allCorrelations = calculateAllCorrelations2D(
        abundancesRIN,
        paramValuesRIN,
        [param],
        confidence_level=0.95,
        calculateConfidenceInterval=True,
    )

    for j, time in enumerate(times):
        plt.scatter(
            parametersDF[param].values,
            abundancesSamples[:, j],
            c=colors[j],
            alpha=alpha,
            s=markersize,
            marker=".",
            # label=f"$r_{{\\mathrm{{RIN}}}}={{{round(allCorrelations['statistic'].iloc[0][j], 2)}}}\pm{{{round(allCorrelations['cidiffdown'].iloc[0][j], 2)}}}$",
            label=f"${{{round(allCorrelations['statistic'].iloc[0][j], 2)}}}\pm{{{round(allCorrelations['cidiffdown'].iloc[0][j], 2)}}}$",
        )

    leg = plt.legend(
        handlelength=0,
        handletextpad=0,
        bbox_to_anchor=(0.975, 0.025),
        loc="lower right",
        alignment="right",
        labelcolor=[colors[i] for i in range(3)],
    )
    [handle.set_visible(False) for handle in leg.legend_handles]
    [t.set_ha("right") for t in leg.get_texts()]

    plt.ylabel("Abundance (wrt H)")
    plt.yscale("log")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.xlabel(convertParameterNameToAxisLabel(param))
    plt.show()


if False:
    UV = 1.0
    zeta = 1.0
    temps = [10.0, 30.0, 50.0]
    densities = [1e3, 1e6]
    # species = ["CH3CHO", "HCOOCH3", "CH3OCH3", "CH3CN", "NH2CHO"]

    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    on_grain = True
    quiet = True
    times = np.logspace(0, 6, num=100)

    paramTypes = ["diff", "diffprefac", "bind", "desprefac", "LH"]
    paramTypesLegend = [
        r"$E_{\mathrm{diff}}$",
        r"$\nu_{\mathrm{diff}}$",
        r"$E_{\mathrm{bind}}$",
        r"$\nu_{\mathrm{des}}$",
        r"$E_{\mathrm{reac}}$",
    ]
    ls = ["solid", "dashdot", "dashed", "dotted", (0, (5, 1))]
    suptitle_h_loc = 0.55

    fig = plt.figure(figsize=(8, 8))
    outer_gs = fig.add_gridspec(len(temps), len(densities), hspace=0.0, wspace=0.0)
    # subfigures = fig.subfigures(3, 2, wspace=0.1, hspace=0.0)
    seenSpeciesParameters = []
    colorsUsed = []

    axsStorage = np.empty(
        shape=(len(temps), len(densities), 2, len(species)), dtype=object
    )
    suplabels = []
    for i, temp in enumerate(temps):
        for j, density in enumerate(densities):
            seenParameters = []
            log10Density = int(np.log10(density))
            title = (
                # letters[i * len(densities) + j] + ") "+
                f"$T={int(temp)}$ K, $n_{{\mathrm{{H}}}}=10^{{{log10Density}}}$ cm$^{{-3}}$"
            )
            subfig = fig.add_subfigure(outer_gs[i, j])
            subfig.suptitle(title, y=0.965, fontsize=12, x=suptitle_h_loc)
            gs = subfig.add_gridspec(
                nrows=2,
                ncols=len(species),
                wspace=0,
                hspace=0,
                height_ratios=(1.0, 2.0 / 3.0),
            )
            gs.update(right=0.975, left=0.125, bottom=0.1, top=0.9)

            axs = np.empty(shape=(2, len(species)), dtype=object)
            for k, spec in enumerate(species):
                if k == 0:
                    axs[0, k] = subfig.add_subplot(gs[0, k])
                    axs[1, k] = subfig.add_subplot(gs[1, k], sharex=axs[0, k])
                else:
                    axs[0, k] = subfig.add_subplot(gs[0, k], sharey=axs[0, 0])
                    axs[1, k] = subfig.add_subplot(
                        gs[1, k], sharey=axs[1, 0], sharex=axs[0, k]
                    )
                axs[0, k].label_outer()
                axs[1, k].label_outer()
                axs[0, k].set_xscale("log")
                axs[0, k].set_xlim(PLOT_XLIM)
                axs[0, k].set_ylim(PLOT_YLIM)
                axs[0, k].set_yscale("log")
                axs[0, k].set_title(convertSpeciesToLegendLabel(spec), y=0.805)

                axs[1, k].set_ylim([-1.2, 1.2])
                axs[1, k].axhline(
                    0,
                    c=ZERO_CORRELATION_COLOR,
                    alpha=ZERO_CORRELATION_ALPHA,
                    ls=ZERO_CORRELATION_LS,
                )
                axs[1, k].fill_between(
                    [0, 1e10],
                    [-MIN_STATISTIC] * 2,
                    [MIN_STATISTIC] * 2,
                    color=WEAK_CORRELATION_COLOR,
                    alpha=WEAK_CORRELATION_ALPHA,
                    edgecolor="none",
                )
            axs[0, 0].set_ylabel("Abundance (wrt H)")
            axs[1, 0].set_ylabel(r"$r_{\mathrm{RIN}}$")
            suplabel = subfig.supxlabel("Time (years)", x=suptitle_h_loc, y=-0.015)
            suplabels.append(suplabel)

            axs[0, 0].set_xticks([1e0, 1e2, 1e4, 1e6])
            [axs[0, l].set_xticks([1e2, 1e4, 1e6]) for l in range(1, len(species))]
            [
                axs[0, l].set_xticks([1e1, 1e3, 1e5], ["", "", ""], minor=True)
                for l in range(len(species))
            ]
            axsStorage[i, j, :, :] = axs

            physicalParamIndex = physicalParamSetToIndex(
                physicalParameterSets,
                T=temp,
                nH=density,
                zeta=zeta,
                radfield=UV,
            )
            physicalParams = physicalParameterSets[physicalParamIndex]
            print(f"Running analysis on {physicalParamSetToString(physicalParams)}")

            sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
                physicalParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
            )

            sampleDFs, correctIndices = removeModelsThatDontReachDesiredTime(
                sampleDFs, times[-1]
            )

            # Because the dfs may not have the same timesteps in the files,
            # calculate at which index the times occur above for each file.
            timeIndices = getTimeIndices(sampleDFs, times)

            # Also get the times that those indices correspond to, such that we can plot them later.
            timeArray = getTimeFromIndices(sampleDFs, timeIndices)

            # Check timeIndices and timeArray, remove timeIndices that correspond to duplicate values of timeArray
            timeIndices, timeArray = checkTimeIndices(
                sampleDFs, timeArray, timeIndices, times, quiet=quiet
            )
            for k, spec in enumerate(species):
                print(f"  Analyzing species {spec}")

                # Lightly plot all model runs
                plotDataFrameAbundances(
                    sampleDFs, spec, on_grain=on_grain, ax=axs[0, k]
                )

                # Also plot nominal model (model with standard network)
                axs[0, k].plot(
                    nominalDF["Time"],
                    nominalDF["#" + spec] + nominalDF["@" + spec],
                    c=NOMINAL_COLOR,
                    ls=NOMINAL_LINESTYLE,
                )

                # Get the abundances of current species at all timesteps in all model runs
                abundancesSamples = getAbundances(
                    sampleDFs, timeIndices, spec=spec, on_grain=on_grain, njobs=N_JOBS
                )
                logMeanAbundance = getLogMeanOfAbundances(abundancesSamples)
                axs[0, k].plot(
                    timeArray[0, :],
                    logMeanAbundance,
                    c=AVERAGE_COLOR,
                    ls=AVERAGE_LINESTYLE,
                )

                # Perform rank-based inverse normal transformation on the abundances
                abundancesRIN = rankInverseNormalTransform(abundancesSamples)
                sigCorrelations = calculateSignificantCorrelations2D(
                    abundancesRIN,
                    parametersRIN[correctIndices, :],
                    parameterNames,
                    confidence_level=0.95,
                    minStatistic=MIN_STATISTIC,
                )

                if sigCorrelations is None:
                    # If there are no strong enough correlations, go to next species
                    continue

                # For all significant and strong correlations, plot them.
                for l, row in sigCorrelations.iterrows():
                    plot_kwargs = {}
                    if row["parameter"] not in seenParameters:
                        seenParameters.append(row["parameter"])
                        plot_kwargs["label"] = convertParameterNameToLegendLabel(
                            row["parameter"]
                        )
                    if "LH" in row["parameter"]:
                        lineStyleIndex = 4
                        speciesParameter = convertParameterNameToLegendLabel(
                            row["parameter"]
                        )
                    else:
                        lineStyleIndex = paramTypes.index(row["parameter"].split()[1])
                        speciesParameter = convertSpeciesToLegendLabel(
                            row["parameter"].split()[0]
                        )

                    if speciesParameter not in seenSpeciesParameters:
                        print(row["parameter"], paramTypes, lineStyleIndex)
                        seenSpeciesParameters.append(speciesParameter)
                    colorIndex = (seenSpeciesParameters.index(speciesParameter)) % len(
                        colors
                    )

                    axs[1, k].plot(
                        sampleDFs[0]["Time"].iloc[timeIndices[0, :]],
                        row["statistic"],
                        c=colors[colorIndex],
                        ls=ls[lineStyleIndex],
                        **plot_kwargs,
                    )

                    # axs[1, k].text(
                    #     1e1,
                    #     0.7,
                    #     convertParameterNameToLegendLabel(row["parameter"]),
                    #     c=colors[colorIndex],
                    # )

                    axs[1, k].fill_between(
                        sampleDFs[0]["Time"].iloc[timeIndices[0, :]],
                        np.array(row["cilow"]) - SAMPLING_95_CI,
                        np.array(row["cihigh"]) + SAMPLING_95_CI,
                        alpha=CORRELATION_CI_ALPHA,
                        color=colors[colorIndex],
                        edgecolor=CORRELATION_CI_EDGECOLOR,
                        linewidth=CORRELATION_CI_LINEWIDTH,
                    )
                # labelLines(
                #     [
                #         line
                #         for line in axs[1, k].get_lines()
                #         if line.get_label()[0] != "_"
                #     ],
                #     align=False,
                #     outline_width=2.5,
                #     fontsize=8.5,
                # )
                # my_legend(axis=axs[1, k])
                # add_inline_labels(
                #     axs[1, k],
                #     with_overall_progress=True,
                #     fontsize="small",
                #     path_effects=[
                #         patheffects.Stroke(linewidth=2.5, foreground="white"),
                #         patheffects.Normal(),
                #     ],
                # )
                # axs[1, k].legend(
                #        handlelength=0,
                #        labelcolor="linecolor",
                #        handletextpad=0,
                #        labelspacing=0.4,
                # )
    legendHandles = [
        Line2D([0], [0], c=colors[(i) % len(colors)])
        for i in range(len(seenSpeciesParameters))
    ]
    leg = axsStorage[0, 0, 0, -1].legend(
        legendHandles,
        [i for i in seenSpeciesParameters],
        loc="center right",
        bbox_to_anchor=(1.0, 1.27),
        handlelength=0,
        labelcolor="linecolor",
        handletextpad=0,
        labelspacing=0.4,
        ncols=6,
    )

    [handle.set_visible(False) for handle in leg.legend_handles]
    axsStorage[0, 0, 0, -1].add_artist(leg)
    leg.set_in_layout(False)

    paramtypeHandles = [Line2D([0], [0], ls=ls[i], c="k") for i in range(5)]
    leg2 = axsStorage[0, 1, 0, 0].legend(
        paramtypeHandles,
        paramTypesLegend,
        loc="center left",
        ncols=3,
        bbox_to_anchor=(0.0, 1.27),
    )
    axsStorage[0, 1, 0, 0].add_artist(leg2)
    leg2.set_in_layout(False)

    # fig.subplots_adjust(left=0.0, bottom=0.1, right=0.99, top=0.8)
    plt.savefig(f"BIGplot_{zeta}_{UV}.pdf", bbox_extra_artists=[leg, leg2, *suplabels])
    plt.show()

if False:
    spec = "C"
    param = "#SO2 diff"
    on_grain = True
    only_surf = False
    only_bulk = True
    T = 50.0
    nH = 1e3
    zeta = 1.0
    UV = 1.0
    time = 1e6

    physicalParamIndex = physicalParamSetToIndex(
        physicalParameterSets,
        T=T,
        nH=nH,
        zeta=zeta,
        radfield=UV,
    )
    physicalParams = physicalParameterSets[physicalParamIndex]
    print(f"Running analysis on {physicalParamSetToString(physicalParams)}")
    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physicalParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
    )

    sampleDFs, correctIndices = removeModelsThatDontReachDesiredTime(sampleDFs, time)
    timeIndices = getTimeIndices(sampleDFs, time)

    abundancesSamples = getAbundances(
        sampleDFs,
        timeIndices,
        spec,
        on_grain=on_grain,
        njobs=N_JOBS,
        only_surf=only_surf,
        only_bulk=only_bulk,
    )

    timeIndexNominal = getTimeIndices([nominalDF], time)[0]
    abundanceNominal = getAbundance(
        nominalDF,
        timeIndexNominal,
        spec,
        on_grain=on_grain,
        only_surf=only_surf,
        only_bulk=only_bulk,
    )

    plt.figure()
    plt.scatter(parametersDF[param].values, abundancesSamples, alpha=0.25, s=5)
    plt.yscale("log")

    plt.show()


if False:
    # Create sensitivity and abundance plots of every single set of physical conditions
    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    # species = ["CH3CHO", "HCOOCH3", "CH3OCH3", "CH3CN", "NH2CHO"]
    #  species = ["N", "NH", "NH3", "N2", "NO"]
    on_grain = True
    times = np.logspace(0, 6, num=100)
    forceCreation = True

    plt.switch_backend("agg")

    temps = [10.0, 20.0, 30.0, 40.0, 50.0]
    densities = [1e3, 1e4, 1e5, 1e6]
    UVs = [0.1, 1.0, 10.0]
    zetas = [0.1, 1.0, 10.0, 100.0]
    UVs = [1.0]
    zetas = [1.0]

    temps = [10.0]
    densities = [1e5]
    UVs = [1.0]
    zetas = [1.0]

    for temp in temps:
        for density in densities:
            for zeta in zetas:
                for UV in UVs:
                    if [temp, density, zeta, UV] not in physicalParameterSets:
                        print(
                            f"Skipping {physicalParamSetToString([temp, density, zeta, UV])}"
                        )
                        continue
                    physicalParamIndex = physicalParamSetToIndex(
                        physicalParameterSets,
                        T=temp,
                        nH=density,
                        zeta=zeta,
                        radfield=UV,
                    )
                    physicalParams = physicalParameterSets[physicalParamIndex]
                    print(
                        f"Running analysis on {physicalParamSetToString(physicalParams)}"
                    )
                    imageSaveString = f"sensitivities_{physicalParamSetToSaveString(physicalParams)}.pdf"
                    if not (
                        forceCreation
                        or not fileIsNewerThanPythonChanges(__file__, imageSaveString)
                    ):
                        print(
                            "Figure path has been modified more recently than python file. Skipping"
                        )
                        continue
                    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
                        physicalParamIndex,
                        filepathsSamples,
                        filepathsNominal,
                        njobs=N_JOBS,
                    )

                    sampleDFs, correctIndices = removeModelsThatDontReachDesiredTime(
                        sampleDFs, times[-1]
                    )
                    parametersRINCorrect = parametersRIN[correctIndices, :]
                    createStandardPlot(
                        sampleDFs,
                        nominalDF,
                        species,
                        parametersRINCorrect,
                        parameterNames,
                        times=times,
                        on_grain=on_grain,
                        minStatistic=MIN_STATISTIC,
                        njobs=N_JOBS,
                        samplingCIError=SAMPLING_95_CI,
                        savefigPath=imageSaveString,
                        doRINtransform=True,
                        parameters=parametersDF.values,
                    )

if False:
    species = ["H2O", "CO", "CO2", "CH3OH"]
    on_grain = True

    temp = 10.0
    densities = [1e3, 1e4, 1e5, 1e6]
    UV = 1.0
    zeta = 1.0

    times = np.logspace(0, 6, num=20)
    sampleCIError = 0.08

    fig, axs = plt.subplots(
        2,
        len(species),
        figsize=(4.5, 3),
        sharex="col",
        sharey="row",
        height_ratios=(1.0, 2.0 / 3.0),
    )
    ls = ["dotted", "-.", "dashed", "solid"]
    hatches = ["/", "\\", "-", "|"]
    seenParameters = []
    for i, spec in enumerate(species):
        axs[1, i].fill_between(
            [0, 1e10],
            [-MIN_STATISTIC] * 2,
            [MIN_STATISTIC] * 2,
            color="k",
            alpha=0.1,
            edgecolor="none",
        )
        axs[1, i].plot([0, 1e10], [0, 0], c="gray", ls="dashed")
        axs[0, i].set_title(convertSpeciesToLegendLabel(spec))

        for j, density in enumerate(densities):
            physicalParamIndex = physicalParamSetToIndex(
                physicalParameterSets,
                T=temp,
                nH=density,
                zeta=zeta,
                radfield=UV,
            )
            physicalParams = physicalParameterSets[physicalParamIndex]
            print(f"Running analysis on {physicalParamSetToString(physicalParams)}")
            sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
                physicalParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
            )

            sampleDFs, correctIndices = removeModelsThatDontReachDesiredTime(
                sampleDFs, times[-1]
            )
            parametersRINCorrect = parametersRIN[correctIndices, :]

            timeIndices = getTimeIndices(sampleDFs, times)
            timeArray = getTimeFromIndices(sampleDFs, timeIndices)
            timeIndices, timeArray = checkTimeIndices(
                sampleDFs, timeArray, timeIndices, times
            )
            abundancesSamples = getAbundances(
                sampleDFs, timeIndices, spec, on_grain=on_grain, njobs=N_JOBS
            )

            axs[0, i].plot(
                nominalDF["Time"],
                nominalDF["#" + spec] + nominalDF["@" + spec],
                c=colors[1],
                ls=ls[j],
            )

            logMeanAbundance = getLogMeanOfAbundances(abundancesSamples)
            axs[0, i].plot(timeArray[0, :], logMeanAbundance, ls=ls[j], c=colors[0])

            abundCIlow, abundCIhigh = getConfidenceIntervalsOfAbundances(
                abundancesSamples, confidence_level=0.6827
            )
            axs[0, i].fill_between(
                timeArray[0, :],
                abundCIlow,
                abundCIhigh,
                color=colors[0],
                alpha=0.25,
                edgecolor="none",
                hatch=hatches[j],
            )

            abundancesRIN = rankInverseNormalTransform(abundancesSamples)
            sigCorrelations = calculateSignificantCorrelations2D(
                abundancesRIN,
                parametersRINCorrect,
                parameterNames,
                minStatistic=MIN_STATISTIC,
                confidence_level=0.95,
            )
            if sigCorrelations is None:
                continue
            for k, row in sigCorrelations.iterrows():
                plot_kwargs = {}
                if row["parameter"] not in seenParameters:
                    seenParameters.append(row["parameter"])
                    plot_kwargs["label"] = convertParameterNameToLegendLabel(
                        row["parameter"]
                    )
                colorIndex = (seenParameters.index(row["parameter"]) + 2) % len(colors)
                axs[1, i].plot(
                    timeArray[0, :],
                    row["statistic"],
                    c=colors[colorIndex],
                    **plot_kwargs,
                    ls=ls[j],
                )
                axs[1, i].fill_between(
                    timeArray[0, :],
                    np.array(row["cilow"]) - sampleCIError,
                    np.array(row["cihigh"]) + sampleCIError,
                    color=colors[colorIndex],
                    alpha=CORRELATION_CI_ALPHA,
                    edgecolor=CORRELATION_CI_EDGECOLOR,
                    linewidth=CORRELATION_CI_LINEWIDTH,
                    hatch=hatches[j],
                )
        axs[1, i].legend()

    fig.supxlabel("Time (years)")

    for ax in axs.flat:
        ax.set_xlim([1e0, 1e6])
        ax.set_xscale("log")
        ax.label_outer()
    # Set xticks
    axs[0, 0].set_xticks([1e0, 1e2, 1e4, 1e6])
    [axs[0, i].set_xticks([1e2, 1e4, 1e6]) for i in range(1, len(species))]
    [
        axs[0, i].set_xticks([1e1, 1e3, 1e5], ["", "", ""], minor=True)
        for i in range(len(species))
    ]

    plt.subplots_adjust(hspace=0, wspace=0, bottom=0.115, left=0.15)
    axs[1, 0].set_ylim([-1.2, 1.2])
    axs[1, 0].set_ylabel(r"$r_{\mathrm{RIN}}$")
    axs[0, 0].set_ylabel("Abundance (wrt H)")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylim([1e-14, 1e-2])
    plt.show()

if False:
    species = ["H2O", "CO", "CO2", "CH3OH"]
    on_grain = True

    temp = 10.0
    densities = [1e3, 1e4, 1e5, 1e6]
    UV = 1.0
    zeta = 1.0

    times = np.logspace(0, 6, num=20)
    sampleCIError = 0.08

    fig, axs = plt.subplots(
        len(species),
        len(densities),
        figsize=(4.5, 3),
        sharex="col",
        sharey="row",
    )
    ls = ["dotted", "-.", "dashed", "solid"]
    # hatches = ["/", "\\", "-", "|"]
    seenParameters = []

    twinax = np.empty(shape=np.shape(axs), dtype=object)
    for j, density in enumerate(densities):
        for i, spec in enumerate(species):
            twinax[i, j] = axs[i, j].twinx()
            if j == 0:
                continue
            else:
                twinax[i, j].sharey(twinax[i, 0])

    for j, density in enumerate(densities):
        physicalParamIndex = physicalParamSetToIndex(
            physicalParameterSets,
            T=temp,
            nH=density,
            zeta=zeta,
            radfield=UV,
        )
        physicalParams = physicalParameterSets[physicalParamIndex]
        print(f"Running analysis on {physicalParamSetToString(physicalParams)}")
        sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
            physicalParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
        )

        sampleDFs, correctIndices = removeModelsThatDontReachDesiredTime(
            sampleDFs, times[-1]
        )
        parametersRINCorrect = parametersRIN[correctIndices, :]

        timeIndices = getTimeIndices(sampleDFs, times)
        timeArray = getTimeFromIndices(sampleDFs, timeIndices)
        timeIndices, timeArray = checkTimeIndices(
            sampleDFs, timeArray, timeIndices, times
        )
        axs[0, j].set_title(convertDensityToTitle(density))
        for i, spec in enumerate(species):
            axs[i, j].text(1e0, 0.9, convertSpeciesToLegendLabel(spec))
            axs[i, j].fill_between(
                [0, 1e10],
                [-MIN_STATISTIC] * 2,
                [MIN_STATISTIC] * 2,
                color="k",
                alpha=0.1,
                edgecolor="none",
            )

            axs[i, j].plot([0, 1e10], [0, 0], c="gray", ls="dashed")
            axs[i, j].set_ylim([-1.2, 1.2])

            abundancesSamples = getAbundances(
                sampleDFs, timeIndices, spec, on_grain=on_grain, njobs=N_JOBS
            )

            abundCIlow, abundCIhigh = getConfidenceIntervalsOfAbundances(
                abundancesSamples, confidence_level=0.6827
            )
            width = abundCIhigh / abundCIlow
            twinax[i, j].plot(
                timeArray[0, :],
                width,
                color="k",
                alpha=0.75,
                ls="dashed",
            )

            abundancesRIN = rankInverseNormalTransform(abundancesSamples)
            sigCorrelations = calculateSignificantCorrelations2D(
                abundancesRIN,
                parametersRINCorrect,
                parameterNames,
                minStatistic=MIN_STATISTIC,
                confidence_level=0.95,
            )
            if sigCorrelations is None:
                continue
            for k, row in sigCorrelations.iterrows():
                plot_kwargs = {}
                if row["parameter"] not in seenParameters:
                    seenParameters.append(row["parameter"])
                    plot_kwargs["label"] = convertParameterNameToLegendLabel(
                        row["parameter"]
                    )
                colorIndex = (seenParameters.index(row["parameter"]) + 2) % len(colors)
                axs[i, j].plot(
                    timeArray[0, :],
                    row["statistic"],
                    c=colors[colorIndex],
                    **plot_kwargs,
                    # ls=ls[j],
                )
                axs[i, j].fill_between(
                    timeArray[0, :],
                    np.array(row["cilow"]) - sampleCIError,
                    np.array(row["cihigh"]) + sampleCIError,
                    color=colors[colorIndex],
                    alpha=CORRELATION_CI_ALPHA,
                    edgecolor=CORRELATION_CI_EDGECOLOR,
                    linewidth=CORRELATION_CI_LINEWIDTH,
                    # hatch=hatches[j],
                )
            axs[i, j].legend()

    fig.supxlabel("Time (years)")

    for ax in axs.flat:
        ax.set_xlim([1e0, 1e6])
        ax.set_xscale("log")

    for j, density in enumerate(densities):
        for i, spec in enumerate(species):
            twinax[i, j].set_yscale("log")
            twinax[i, j].label_outer()

    # Set xticks
    axs[0, 0].set_xticks([1e0, 1e2, 1e4, 1e6])
    [axs[0, i].set_xticks([1e2, 1e4, 1e6]) for i in range(1, len(densities))]
    [
        axs[0, i].set_xticks([1e1, 1e3, 1e5], ["", "", ""], minor=True)
        for i in range(len(species))
    ]

    plt.subplots_adjust(hspace=0, wspace=0, bottom=0.115, left=0.15)
    # axs[0, 0].set_ylim([-1.2, 1.2])
    axs[0, 0].set_ylabel(r"$r_{\mathrm{RIN}}$")
    plt.show()

if False:
    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    # species = ["CH3CHO", "HCOOCH3", "CH3OCH3", "CH3CN", "NH2CHO"]
    on_grain = True

    temps = [10, 20, 30, 40, 50]
    densities = [1e3, 1e4, 1e5, 1e6]
    UV = 1.0
    zeta = 1.0
    confidence_level = 0.6827

    times = np.logspace(0, 6, num=100)

    fig, axs = plt.subplots(
        1,
        len(species),
        figsize=(4.5, 2.598425197),
        sharex=False,
        sharey=True,
    )
    widthsAtDensities = np.zeros((len(species), len(densities), len(times)))

    temp_colors = sns.color_palette("flare_r", as_cmap=True)
    norm = mpl.colors.BoundaryNorm([5, 15, 25, 35, 45, 55], temp_colors.N)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=temp_colors)

    for k, temp in enumerate(temps):
        for j, density in enumerate(densities):
            physicalParamIndex = physicalParamSetToIndex(
                physicalParameterSets,
                T=temp,
                nH=density,
                zeta=zeta,
                radfield=UV,
            )
            physicalParams = physicalParameterSets[physicalParamIndex]
            print(f"Running analysis on {physicalParamSetToString(physicalParams)}")
            sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
                physicalParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
            )

            sampleDFs, correctIndices = removeModelsThatDontReachDesiredTime(
                sampleDFs, times[-1]
            )
            parametersRINCorrect = parametersRIN[correctIndices, :]

            timeIndices = getTimeIndices(sampleDFs, times)
            timeArray = getTimeFromIndices(sampleDFs, timeIndices)
            timeIndices, timeArray = checkTimeIndices(
                sampleDFs, timeArray, timeIndices, times
            )
            for i, spec in enumerate(species):
                abundancesSamples = getAbundances(
                    sampleDFs, timeIndices, spec, on_grain=on_grain, njobs=N_JOBS
                )

                abundCIlow, abundCIhigh = getConfidenceIntervalsOfAbundances(
                    abundancesSamples, confidence_level=confidence_level
                )
                width = abundCIhigh / abundCIlow
                width = np.log10(width)
                widthsAtDensities[i, j, : len(width)] = width
                # axs[i].plot(
                #     timeArray[0, :],
                #     width,
                #     c=temp_colors(k*1./len(temps)),
                #     alpha=0.2,
                #     linewidth=0.5,
                # )

        for i, spec in enumerate(species):
            averageWidthOverDensities = np.average(
                widthsAtDensities[i, :, : len(width)], axis=0
            )
            axs[i].plot(
                timeArray[0, :],
                np.power(10, averageWidthOverDensities),
                c=temp_colors(k * 1.0 / len(temps)),
                zorder=2,
            )

    for i, spec in enumerate(species):
        axs[i].set_title(convertSpeciesToLegendLabel(spec), y=0.975)

        # Draw line at 2 orders of magnitude.
        axs[i].axhline(1e2, c="gray", ls="dashed", alpha=0.6, zorder=1)

    fig.supxlabel("Time (years)", x=0.49, y=0.0)
    for ax in axs.flat:
        ax.set_xlim([1e0, 1e6])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.label_outer()
    # Set xticks
    axs[0].set_xticks([1e0, 1e2, 1e4, 1e6])
    [axs[i].set_xticks([1e2, 1e4, 1e6]) for i in range(1, len(species))]
    [
        axs[i].set_xticks([1e1, 1e3, 1e5], ["", "", ""], minor=True)
        for i in range(len(species))
    ]

    # Set bottom of plot to 1
    axs[i].set_ylim(bottom=1e0)

    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.83, 0.115, 0.025, 0.765])
    cb = fig.colorbar(sm, ticks=temps, cax=cbar_ax)
    cb.ax.tick_params(axis="y", direction="out")
    cb.ax.minorticks_off()
    cb.set_label("Temperature (K)")

    plt.subplots_adjust(hspace=0, wspace=0, bottom=0.115, left=0.15)
    axs[0].set_ylabel(f"Width of {confidence_level * 100}\% confidence interval")
    # axs[0].set_yscale("log")
    plt.savefig("widths_order_of_magnitude_COMs.pdf")
    plt.show()

if False:
    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    times = np.logspace(0, 6, num=50)
    on_grain = True
    T = 20.0
    zeta = 1.0
    UV = 1.0
    density = 1e5

    physParamIndex = physicalParamSetToIndex(
        physicalParameterSets, T=T, nH=density, zeta=zeta, radfield=UV
    )
    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
    )

    sampleDFs, correctIndices = removeModelsThatDontReachDesiredTime(
        sampleDFs, times[-1]
    )

    timeIndices = getTimeIndices(sampleDFs, times)
    timeArray = getTimeFromIndices(sampleDFs, timeIndices)
    timeIndices, timeArray = checkTimeIndices(sampleDFs, timeArray, timeIndices, times)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=False)
    axs[0].set_xlim([1e0, 1e6])
    axs[0].set_xscale("log")
    [axs[i].set_yscale("log") for i in range(2)]
    [axs[i].set_xlabel("Time (years)") for i in range(2)]
    axs[0].set_ylabel("Abundance (wrt H)")
    axs[1].set_ylabel("Width of 67th percentile of abundance (wrt H)")
    for i, spec in enumerate(species):
        abundancesSamples = getAbundances(
            sampleDFs, timeIndices, spec, on_grain=on_grain, njobs=N_JOBS
        )

        logMeanAbundance = getLogMeanOfAbundances(abundancesSamples)
        axs[0].plot(
            nominalDF["Time"],
            nominalDF["#" + spec] + nominalDF["@" + spec],
            c=colors[i],
            ls="dashed",
        )
        axs[0].plot(
            timeArray[0, :],
            logMeanAbundance,
            c=colors[i],
            label=convertSpeciesToLegendLabel(spec),
        )

        CIlow, CIhigh = getConfidenceIntervalsOfAbundances(
            abundancesSamples, confidence_level=0.6827
        )
        axs[0].fill_between(
            timeArray[0, :],
            CIlow,
            CIhigh,
            color=colors[i],
            edgecolor="none",
            alpha=0.25,
        )

        width = CIhigh / CIlow
        axs[1].plot(timeArray[0, :], width)
    plt.show()

if False:
    spec = "OH"
    on_grain = True
    T = 10.0
    zeta = 1.0
    UV = 10.0

    densities = [1e3, 1e4, 1e5, 1e6]
    times = np.logspace(0, 6, num=20)

    seenParameters = []
    fig, axs = plt.subplots(2, 4, sharex="col", sharey="row", figsize=(4.5, 3))
    for i, density in enumerate(densities):
        axs[0, i].set_title(f"{density} cm$^{{-3}}$")
        axs[0, i].set_xlabel("Time (years)")

        physParamIndex = physicalParamSetToIndex(
            physicalParameterSets, T=T, nH=density, zeta=zeta, radfield=UV
        )
        sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
            physParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
        )

        sampleDFs = removeModelsThatDontReachDesiredTime(sampleDFs, times[-1])
        plotDataFrameAbundances(sampleDFs, spec, on_grain=on_grain, ax=axs[0, i])

        axs[0, i].plot(
            nominalDF["Time"],
            nominalDF["#" + spec] + nominalDF["@" + spec],
            c=colors[1],
        )

        timeIndices = getTimeIndices(sampleDFs, times)
        timeArray = getTimeFromIndices(sampleDFs, timeIndices)
        timeIndices, timeArray = checkTimeIndices(
            sampleDFs, timeArray, timeIndices, times
        )
        abundancesSamples = getAbundances(
            sampleDFs, timeIndices, spec, on_grain=on_grain, njobs=N_JOBS
        )
        abundancesRIN = rankInverseNormalTransform(abundancesSamples)

        # Add horizontal line at correlation = 0, and fill areas where correlation < minStatistic
        axs[1, i].plot([0, 1e10], [0] * 2, color="gray", ls="dashed")
        axs[1, i].fill_between(
            [0, 1e10],
            [-MIN_STATISTIC] * 2,
            [MIN_STATISTIC] * 2,
            color="k",
            alpha=0.1,
            edgecolor="none",
        )

        sigCorrelations = calculateSignificantCorrelations2D(
            abundancesRIN,
            parametersRIN,
            parameterNames,
            confidence_level=0.95,
            minStatistic=MIN_STATISTIC,
        )

        if sigCorrelations is None:
            continue

        # For all significant and strong correlations, plot them.
        for j, row in sigCorrelations.iterrows():
            if row["parameter"] not in seenParameters:
                seenParameters.append(row["parameter"])
            colorIndex = (seenParameters.index(row["parameter"]) + 2) % len(colors)
            axs[1, i].plot(
                sampleDFs[0]["Time"].iloc[timeIndices[0, :]],
                row["statistic"],
                label=convertParameterNameToLegendLabel(row["parameter"]),
                c=colors[colorIndex],
            )

            axs[1, i].fill_between(
                sampleDFs[0]["Time"].iloc[timeIndices[0, :]],
                row["cilow"],
                row["cihigh"],
                alpha=0.25,
                color=colors[colorIndex],
                edgecolor="none",
            )

        axs[1, i].legend()

    for ax in axs.flat:
        ax.set_xlim([1e0, 1e6])
        ax.set_xscale("log")
        ax.label_outer()

    axs[0, 0].set_ylim([1e-14, 1e-2])
    axs[0, 0].set_yscale("log")
    axs[1, 0].set_ylim([-1.2, 1.2])
    axs[1, 0].set_yticks([-1, 0, 1])
    axs[0, 0].set_ylabel("Abundance (wrt H)")
    axs[1, 0].set_ylabel(r"$r_{\mathrm{RIN}}$")
    # [axs[1, i].set_xlabel("Time (years)") for i in range(len(species))]
    fig.supxlabel("Time (years)")

    # Set xticks
    axs[0, 0].set_xticks([1e0, 1e2, 1e4, 1e6])
    [axs[0, i].set_xticks([1e2, 1e4, 1e6]) for i in range(1, len(axs[0, :]))]
    [
        axs[0, i].set_xticks([1e1, 1e3, 1e5], ["", "", ""], minor=True)
        for i in range(len(axs[0, :]))
    ]

    plt.subplots_adjust(hspace=0, wspace=0.03, bottom=0.15, left=0.15)
    figureFilepath = f"sensitivitiesAtDifferentDens_{spec}-{on_grain}_{T}K_{zeta * 1.3e-17:.1e}s-1_{UV}Habing.pdf"
    plt.savefig(figureFilepath)
    plt.show()

if False:
    spec = "CH3OH"
    on_grain = True
    T = 10.0
    zeta = 1.0

    UVs = [0.1, 1.0, 10.0]
    ls = ["dashed", "dotted", "solid"]
    densities = [1e3, 1e4, 1e5, 1e6]
    times = np.logspace(0, 6, num=20)

    seenParameters = []
    fig, axs = plt.subplots(2, 4, sharex="col", sharey="row", figsize=(4.5, 3))
    for i, density in enumerate(densities):
        axs[0, i].set_title(f"{density} cm$^{{-3}}$")
        axs[0, i].set_xlabel("Time (years)")
        for k, UVstrength in enumerate(UVs):
            physParamIndex = physicalParamSetToIndex(
                physicalParameterSets, T=T, nH=density, zeta=zeta, radfield=UVstrength
            )
            sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
                physParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
            )
            plotDataFrameAbundances(
                sampleDFs,
                spec,
                on_grain=on_grain,
                ax=axs[0, i],
                ls=ls[k],
            )

            axs[0, i].plot(
                nominalDF["Time"],
                nominalDF["#" + spec] + nominalDF["@" + spec],
                c=colors[1],
            )

            timeIndices = getTimeIndices(sampleDFs, times)
            timeArray = getTimeFromIndices(sampleDFs, timeIndices)
            timeIndices, timeArray = checkTimeIndices(
                sampleDFs, timeArray, timeIndices, times
            )
            abundancesSamples = getAbundances(
                sampleDFs, timeIndices, spec, on_grain=True, njobs=N_JOBS
            )
            abundancesRIN = rankInverseNormalTransform(abundancesSamples)

            correlations = calculateAllCorrelations2D(
                abundancesRIN,
                parametersRIN,
                parameterNames,
                calculateConfidenceInterval=False,
            )

            # Find which parameter indices are at any timepoint strong enough and significant
            sigIndices = getSignificantCorrelationsIndices(
                correlations, minStatistic=MIN_STATISTIC
            )

            # Add horizontal line at correlation = 0, and fill areas where correlation < minStatistic
            axs[1, i].plot([0, 1e10], [0] * 2, color="gray", ls="dashed")
            axs[1, i].fill_between(
                [0, 1e10],
                [-MIN_STATISTIC] * 2,
                [MIN_STATISTIC] * 2,
                color="k",
                alpha=0.1,
                edgecolor="none",
            )

            if not sigIndices:
                # If there are none, continue to the next species
                continue

            # Recalculate the strong and significant correlations, now also calculate their confidence intervals
            sigCorrelations = calculateAllCorrelations2D(
                abundancesRIN,
                parametersRIN[:, sigIndices],
                parameterNames[sigIndices],
                calculateConfidenceInterval=True,
                confidence_level=0.95,
            )

            # For all significant and strong correlations, plot them.
            for j, row in sigCorrelations.iterrows():
                if row["parameter"] not in seenParameters:
                    seenParameters.append(row["parameter"])
                colorIndex = (seenParameters.index(row["parameter"]) + 2) % len(colors)
                axs[1, i].plot(
                    sampleDFs[0]["Time"].iloc[timeIndices[0, :]],
                    row["statistic"],
                    label=convertParameterNameToLegendLabel(row["parameter"]),
                    c=colors[colorIndex],
                    ls=ls[k],
                )

                axs[1, i].fill_between(
                    sampleDFs[0]["Time"].iloc[timeIndices[0, :]],
                    row["cilow"],
                    row["cihigh"],
                    alpha=0.25,
                    color=colors[colorIndex],
                    edgecolor="none",
                )

        axs[1, i].legend()

    for ax in axs.flat:
        ax.set_xlim([1e0, 1e6])
        ax.set_xscale("log")
        ax.label_outer()

    axs[0, 0].set_ylim([1e-14, 1e-2])
    axs[0, 0].set_yscale("log")
    axs[1, 0].set_ylim([-1.2, 1.2])
    axs[1, 0].set_yticks([-1, 0, 1])
    axs[0, 0].set_ylabel("Abundance (wrt H)")
    axs[1, 0].set_ylabel(r"$r_{\mathrm{RIN}}$")
    # [axs[1, i].set_xlabel("Time (years)") for i in range(len(species))]
    fig.supxlabel("Time (years)")

    # Set xticks
    axs[0, 0].set_xticks([1e0, 1e2, 1e4, 1e6])
    [axs[0, i].set_xticks([1e2, 1e4, 1e6]) for i in range(1, len(axs[0, :]))]
    [
        axs[0, i].set_xticks([1e1, 1e3, 1e5], ["", "", ""], minor=True)
        for i in range(len(axs[0, :]))
    ]

    plt.subplots_adjust(hspace=0, wspace=0.03, bottom=0.15, left=0.15)
    figureFilepath = f"sensitivitiesAtDifferentDensAndUV_{spec}-{on_grain}_{T}K_{zeta * 1.3e-17:.1e}s-1.pdf"
    plt.savefig(figureFilepath)
    plt.show()


if False:
    spec = "H2O"
    parameter = "#H2O bind"
    on_grain = True
    T = 20.0
    zeta = 1.0
    density = 1e5
    radfield = 1.0
    time = 1e2

    physParamIndex = physicalParamSetToIndex(
        physicalParameterSets, T=T, nH=density, zeta=zeta, radfield=radfield
    )
    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
    )

    timeIndices = getTimeIndices(sampleDFs, time)

    abundancesSamples = getAbundances(
        sampleDFs, timeIndices, spec=spec, on_grain=on_grain
    )

    reactionsFile = "/home/dijkhuis/PhD/UCLCHEM_priv/uclchem/src/uclchem/reactions.csv"
    speciesFile = "/home/dijkhuis/PhD/UCLCHEM_priv/uclchem/src/uclchem/species.csv"

    reactions = pd.read_csv(reactionsFile)
    species = pd.read_csv(speciesFile)

    for i, row in reactions.iterrows():
        if row["Reactant 3"] != "LHDES":
            continue
        if row["Reactant 1"] != "#H":
            continue
        if row["Reactant 2"] != "#OH":
            continue
        print(row)
        chi = calculateChi(row, species)
        reactionEnthalpy = getEnthalpyOfReaction(row, species)
    desorptionProbabilities = [
        getDesorptionProbability(chi, reactionEnthalpy, i)
        for i in parametersDF[parameter].values
    ]
    rankAbundances = rankData(abundancesSamples)
    plt.figure()
    plt.scatter(
        parametersDF[parameter].values, abundancesSamples, c=colors[0], alpha=0.5
    )
    plt.plot(parametersDF[parameter].values, desorptionProbabilities)
    plt.yscale("log")
    plt.show()


if False:
    spec = "H2O"
    parameter = "#H desprefac"
    on_grain = True
    T = 40.0
    zeta = 1.0
    density = 1e3
    radfield = 1.0
    time = 1e4

    physParamIndex = physicalParamSetToIndex(
        physicalParameterSets, T=T, nH=density, zeta=zeta, radfield=radfield
    )
    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
    )

    timeIndices = getTimeIndices(sampleDFs, time)

    abundancesSamples = getAbundances(
        sampleDFs, timeIndices, spec=spec, on_grain=on_grain
    )

    plt.figure()
    plt.scatter(
        parametersDF[parameter].values, abundancesSamples, c=colors[0], alpha=0.5
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

if False:
    specs = ["H", "N", "O"]
    T = 10.0
    zeta = 1.0
    density = 1e5
    radfield = 1.0

    physParamIndex = physicalParamSetToIndex(
        physicalParameterSets, T=T, nH=density, zeta=zeta, radfield=radfield
    )
    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
    )

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    for i, spec in enumerate(specs):
        axs[0].plot(nominalDF["Time"], nominalDF[spec], label=spec, c=colors[i])
        axs[1].plot(nominalDF["Time"], nominalDF["#" + spec], label=spec, c=colors[i])
        axs[2].plot(nominalDF["Time"], nominalDF["@" + spec], label=spec, c=colors[i])
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_ylim([1e-16, 1e-4])
    axs[2].set_xlim([1e0, 1e6])
    axs[2].legend()
    [axs[i].set_title(title) for i, title in enumerate(["Gas", "Surface", "Bulk"])]
    plt.tight_layout()
    plt.show()


if False:
    specs = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    T = 10.0
    zeta = 1.0
    density = 1e6
    radfield = 1.0
    time = 1e6
    on_grain = True

    physParamIndex = physicalParamSetToIndex(
        physicalParameterSets, T=T, nH=density, zeta=zeta, radfield=radfield
    )
    sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
        physParamIndex, filepathsSamples, filepathsNominal, njobs=N_JOBS
    )

    # test_convergence(
    #     sampleDFs, specs, timeIndices, parameters, parameterNames, on_grain=on_grain
    # )

    convergenceDir = "/data2/dijkhuis/ChemSamplingMC/setWidthProductionTightConvergence"
    parametersConvergence = pd.read_csv(
        convergenceDir + "/MC_parameter_runs.csv", index_col=0
    )
    cols = parametersConvergence.columns
    erCol = [col for col in cols if "ER" in col]
    parametersConvergence.drop(labels=erCol, inplace=True, axis=1)
    parametersConvergence = parametersConvergence.to_numpy()

    # Get filepaths of all model runs
    filepathsConvergence, filepathsNominalConvergence = getAllRunsFilepaths(
        convergenceDir, extension="h5"
    )

    # Get all sets of physical conditions
    physicalParameterSetsConvergence = getPhysicalParamSets(
        [filepaths[0] for filepaths in filepathsConvergence]
    )

    physParamIndexConvergence = physicalParamSetToIndex(
        physicalParameterSetsConvergence, T=T, nH=density, zeta=zeta, radfield=radfield
    )
    convergenceDFs, nominalDFConvergence = getDataFramesForPhysicalParamSet(
        physParamIndexConvergence,
        filepathsConvergence,
        filepathsNominalConvergence,
        njobs=N_JOBS,
    )
    parametersRINconvergence = rankInverseNormalTransform(parametersConvergence)
    # createStandardPlot(
    #     sampleDFs,
    #     nominalDF,
    #     specs,
    #     parametersRIN,
    #     parameterNames,
    #     on_grain=on_grain,
    #     minStatistic=MIN_STATISTIC,
    #     samplingCIError=SAMPLING_95_CI,
    #     njobs=N_JOBS,
    #     savefigPath="test_samples.pdf",
    # )
    # createStandardPlot(
    #     convergenceDFs,
    #     nominalDFConvergence,
    #     specs,
    #     parametersRINconvergence,
    #     parameterNames,
    #     on_grain=on_grain,
    #     minStatistic=MIN_STATISTIC,
    #     samplingCIError=SAMPLING_95_CI,
    #     njobs=N_JOBS,
    #     savefigPath="test_convergence.pdf",
    # )

    # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    # for i, spec in enumerate(specs):
    #     axs[0].plot(nominalDF["Time"], nominalDF[spec], c=colors[i])
    #     axs[1].plot(nominalDF["Time"], nominalDF["#" + spec], c=colors[i])
    #     axs[2].plot(nominalDF["Time"], nominalDF["@" + spec], c=colors[i])

    #     axs[0].plot(
    #         nominalDFConvergence["Time"],
    #         nominalDFConvergence[spec],
    #         c=colors[i],
    #         ls="dashed",
    #     )
    #     axs[1].plot(
    #         nominalDFConvergence["Time"],
    #         nominalDFConvergence["#" + spec],
    #         c=colors[i],
    #         ls="dashed",
    #     )
    #     axs[2].plot(
    #         nominalDFConvergence["Time"],
    #         nominalDFConvergence["@" + spec],
    #         c=colors[i],
    #         ls="dashed",
    #     )

    # axs[0].set_xlim([1e0, 1e6])
    # axs[0].set_xscale("log")
    # axs[0].set_yscale("log")
    # plt.show()

    compare_with_converged(
        sampleDFs,
        convergenceDFs,
        specs,
        time,
        parameters,
        parametersConvergence,
        parameterNames,
        on_grain=on_grain,
        njobs=N_JOBS,
        savefig_path="convergence.pdf",
    )

if __name__ == "__main__":
    pass
