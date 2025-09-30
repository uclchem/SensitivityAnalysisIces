import itertools
import math
import os
import re
from copy import deepcopy
from glob import glob
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap

# from labellines import labelLine, labelLines
from scipy import integrate, ndimage, stats
from scipy.constants import atomic_mass, h, hbar, k, pi

# from uclchem.analysis import (_format_reactions,rates_change,
#                               _param_dict_from_output, analysis,
#                               analysis_solid_phase, getNetChange,
#                               read_analysis, read_output_file)
from uclchem.analysis import (
    _format_reactions,
    _param_dict_from_output,
    read_output_file,
)
from uclchem.constants import n_species
from uclchem.makerates.reaction import Reaction
from uclchem.uclchemwrap import uclchemwrap as wrap

GAS_DUST_MASS_RATIO = 100.0e0
GRAIN_RADIUS = 1.0e-5
GRAIN_DENSITY = 3.0e0
GAS_DUST_DENSITY_RATIO = (
    4.0e0 * pi * (GRAIN_RADIUS**3) * GRAIN_DENSITY * GAS_DUST_MASS_RATIO
) / (3.0e0 * atomic_mass)  # Reciprocal of gas-to-dust number density
SURFACE_SITE_DENSITY = 1.5e15
NUM_SITES_PER_GRAIN = GRAIN_RADIUS * GRAIN_RADIUS * SURFACE_SITE_DENSITY * 4.0 * pi


def getDiscreteRainbowColors(ncolors: int, omit_yellow: bool = True) -> list[str]:
    discreteColors = [
        "#E8ECFB",
        "#D9CCE3",
        "#D1BBD7",
        "#CAACCB",
        "#BA8DB4",
        "#AE76A3",
        "#AA6F9E",
        "#994F88",
        "#882E72",
        "#1965B0",
        "#437DBF",
        "#5289C7",
        "#6195CF",
        "#7BAFDE",
        "#4EB265",
        "#90C987",
        "#CAE0AB",
        "#F7F056",
        "#F7CB45",
        "#F6C141",
        "#F4A736",
        "#F1932D",
        "#EE8026",
        "#E8601C",
        "#E65518",
        "#DC050C",
        "#A5170E",
        "#72190E",
        "#42150A",
    ]
    if omit_yellow:
        discreteColors.pop(17)

    # List that indicates which color is changing.
    # Positive integer indicates that a new color is being added,
    # and negative indicates that that color is being removed.
    # Note that this is 1-indexed, as Paul Tol's triangle plot,
    # and python is of course 0-indexed.
    # For example: 18 indicates color 18 is being added, and
    # -18 indicates color 18 is being removed.
    addingOrder = [
        10,
        26,
        18,
        15,
        14,
        17,
        9,
        23,
        28,
        (21, -23, 24),
        12,
        (3, 6, -28),
        16,
        28,
        (5, -6, 7),
        8,
        27,
        (2, -3, 4),
        (11, -12, 13),
        (19, -20, 21, -22, 23, -24, 25),
        29,
        1,
    ]

    colors = []
    for i in range(ncolors):
        index = addingOrder[i]
        if isinstance(index, int):
            # Just add that index to colors
            colors.append(discreteColors[index - 1])
        elif isinstance(index, tuple):
            for j, tupleField in enumerate(index):
                if tupleField < 0:
                    pass
        else:
            raise ValueError()


def arrheniusProb(Ereac: float, temp: float) -> float:
    """Probability of an Arrhenius process

    Inputs:
        Ereac (float): energy barrier in K
        temp (float): temperature in K

    Returns:
        float: probability of barrier being crossed at temperature
    """
    return np.exp(-Ereac / temp)


def rectangularBarrierProb(
    Ereac: float, tunnelingMass: float, barrierWidth: float = 1.4
) -> float:
    """Tunneling probability through symmetric rectangular barrier

    Inputs:
        Ereac (float): energy barrier in K
        tunnelingMass (float): tunneling mass of the reaction in atomic mass units.
            Often approximated as the reduced mass of the two reagents.
        barrierWidth (float): width of barrier in Angstrom. Default: 1.4

    Returns:
        float: tunneling probability
    """
    return np.exp(
        -2.0
        * 1.4
        * 1e-10
        / hbar
        * np.sqrt(2.0 * tunnelingMass * atomic_mass * k * Ereac)
    )


def reactionProb(
    Ereac: list[float] | float, temp: float, tunnelingMass: float, barrierWidth=1.4
) -> list[float] | float:
    """Calculate reaction probability at a certain temperature. Maximum of Arrhenius probability and tunneling probability

    Inputs:
        Ereac (float): energy barrier in K
        temp (float): temperature in K
        tunnelingMass (float): tunneling mass of the reaction in atomic mass units.
            Often approximated as the reduced mass of the two reagents.
        barrierWidth (float): width of barrier in Angstrom. Default: 1.4

    Returns:
        list[float] | float: reaction probability
    """
    if isinstance(Ereac, list) or isinstance(Ereac, np.ndarray):
        reactionProbs = np.empty(len(Ereac))
        for i, E in enumerate(Ereac):
            reactionProbs[i] = max(
                arrheniusProb(E, temp),
                rectangularBarrierProb(E, tunnelingMass, barrierWidth=barrierWidth),
            )
        return reactionProbs
    return max(
        arrheniusProb(Ereac, temp),
        rectangularBarrierProb(Ereac, tunnelingMass, barrierWidth=barrierWidth),
    )


def noCompetitionRateConstant(
    prefac: float,
    Ereac: float,
    tunnelingMass: float,
    temp: float,
    barrierWidth: float = 1.4,
) -> float:
    """Calculate rate constant of reaction while not taking competition into account.

    Inputs:
        prefac (float): prefactor in s-1
        Ereac (float): energy barrier in K
        temp (float): temperature in K
        tunnelingMass (float): tunneling mass of the reaction in atomic mass units.
            Often approximated as the reduced mass of the two reagents.
        barrierWidth (float): width of barrier in Angstrom. Default: 1.4

    Returns:
        float: reaction probability
    """
    return reactionProb(Ereac, temp, tunnelingMass, barrierWidth=barrierWidth) * prefac


def calculateTSTprefactor(
    spec: pd.Series, temp: float, surfaceSiteDensity: float = 1.5e15
) -> float:
    """Calculate the Transition State Theory (TST) prefactor for desorption

    Inputs:
        spec (pd.Series): row of dataframe containing columns 'Ix', 'Iy', 'Iz', 'SYMMETRY FACTOR', and 'MASS'
            'Ix', 'Iy', and 'Iz' are in units of amu angstrom^2.
        temp (float): dust temperature in K
        surfaceSiteDensity (float): site density on grain in cm^-2

    Returns:
        float: Transition State Theory desorption prefactor in s^-1
    """
    lambdaSquared = h**2 / (2.0 * pi * spec["MASS"] * atomic_mass * k * temp)
    qtrans = 1 / (surfaceSiteDensity * 1e4 * lambdaSquared)

    AMU_ANGSTROM2_TO_KG_M2 = atomic_mass * 1e-20
    # convert Ix,  Iy, and Iz to correct units
    Ix = spec["Ix"] * AMU_ANGSTROM2_TO_KG_M2
    Iy = spec["Iy"] * AMU_ANGSTROM2_TO_KG_M2
    Iz = spec["Iz"] * AMU_ANGSTROM2_TO_KG_M2

    symmetryFactor = spec["SYMMETRY FACTOR"]

    if Ix <= 0:
        # Linear
        qrot = 1 / (h**2) * (8 * pi**2 * k * temp) * np.sqrt(Iy * Iz) / symmetryFactor
    else:
        # Non-linear
        qrot = (
            np.sqrt(pi)
            / (h**3)
            * (8 * pi**2 * k * temp) ** (3.0 / 2.0)
            * np.sqrt(Ix * Iy * Iz)
            / symmetryFactor
        )
    return k * temp / h * qtrans * qrot


def calculateHHprefactor(spec: pd.Series, surfaceSiteDensity: float = 1.5e15) -> float:
    """Calculate the Hasegawa-Herbst prefactor for desorption and diffusion

    Inputs:
        spec (pd.Series): row of dataframe containing columns 'BINDING ENERGY' and 'MASS'
        surfaceSiteDensity (float): site density on grain in cm-2

    Returns:
        float: Hasegawa-Herbst prefactor in s-1
    """
    return np.sqrt(
        2.0
        * surfaceSiteDensity
        * (spec["BINDING ENERGY"] * k)
        * 1e4
        / (pi * pi * spec["MASS"] * atomic_mass)
    )


def calculateRateConstant(
    Ereac: float,
    Tdust: float,
    tunnelingMass: float,
    vdiffA: float,
    vdiffB: float,
    EdiffA: float,
    EdiffB: float,
    vdesA: float,
    vdesB: float,
    EbindA: float,
    EbindB: float,
    barrierWidth: float = 1.4,
    only_competition_fraction: bool = False,
    # num_sites_per_grain: float = 1.5e15,
) -> float:
    probThermal = arrheniusProb(Ereac, Tdust)
    probTunneling = rectangularBarrierProb(
        Ereac, tunnelingMass, barrierWidth=barrierWidth
    )
    reacProb = max(probThermal, probTunneling)

    reacRate = max(vdiffA, vdiffB) * reacProb

    diffusionRate = vdiffA * arrheniusProb(EdiffA, Tdust) + vdiffB * arrheniusProb(
        EdiffB, Tdust
    )
    desorptionRate = vdesA * arrheniusProb(EbindA, Tdust) + vdesB * arrheniusProb(
        EbindB, Tdust
    )

    kappaReac = reacRate / (reacRate + diffusionRate + desorptionRate)
    if only_competition_fraction:
        return kappaReac
    # alpha(reacIndx) *reactionProb* diffuseProb*GAS_DUST_DENSITY_RATIO/NUM_SITES_PER_GRAIN

    # Not actual rate constant, just how it scales with the various physical/chemical parameters
    # such as binding energies, prefactors, diffusion barriers, energy barrier and temperature.
    # This will not matter for the correlations, since linear transformations will still result
    # in the same correlation coefficients
    return kappaReac * diffusionRate  # * GAS_DUST_DENSITY_RATIO / NUM_SITES_PER_GRAIN


def calculateRateConstantFromRow(
    row: list[float],
    reaction: str,
    tunnelingMass: float,
    Tdust: float,
    barrierWidth: float = 1.4,
    only_competition_fraction: bool = False,
):
    splitReaction = reaction.split()

    reacA = splitReaction[0]
    EdiffA = row[f"{reacA} diff"]
    vdiffA = row[f"{reacA} diffprefac"]
    EbindA = row[f"{reacA} bind"]
    vdesA = row[f"{reacA} desprefac"]

    reacB = splitReaction[2]
    EdiffB = row[f"{reacB} diff"]
    vdiffB = row[f"{reacB} diffprefac"]
    EbindB = row[f"{reacB} bind"]
    vdesB = row[f"{reacB} desprefac"]

    Ereac = row[reaction]

    rateConstant = calculateRateConstant(
        Ereac,
        Tdust,
        tunnelingMass,
        vdiffA,
        vdiffB,
        EdiffA,
        EdiffB,
        vdesA,
        vdesB,
        EbindA,
        EbindB,
        barrierWidth=barrierWidth,
        only_competition_fraction=only_competition_fraction,
    )

    return rateConstant


def calculateNumberPerGrain(abundance: float):
    """Calculates the number of species per grain from the abundance.

    Inputs:
        abundance (float): abundance with respect to H

    Returns:
        float: number of species per grain
    """
    return abundance * GAS_DUST_DENSITY_RATIO


def calculateRateConstantAllRows(
    parameterDF: pd.DataFrame,
    reaction: str,
    Tdust: float,
    barrierWidth: float = 1.4,
    only_competition_fraction: bool = False,
    tunnelingMass: int | None = None,
):
    splitReaction = reaction.replace("+", "").replace("->", "").split()

    reacA = splitReaction[0]
    reacB = splitReaction[1]

    nprods = len(splitReaction[3:])
    splitReaction += [""] * (4 - nprods)
    splitReaction += [0] * 6

    if tunnelingMass is None:
        tunnelingMass = Reaction(splitReaction).get_reduced_mass()

    rateConstants = np.empty(len(parameterDF.index))
    for i, row in parameterDF.iterrows():
        rateConstants[i] = calculateRateConstantFromRow(
            row,
            reaction,
            tunnelingMass,
            Tdust,
            barrierWidth=barrierWidth,
            only_competition_fraction=only_competition_fraction,
        )
    return rateConstants


def calculateRateConstantsFromParameters(
    parameterDF: pd.DataFrame,
    Tdust: float,
    barrierWidth: float = 1.4,
    only_competition_fraction: bool = False,
) -> pd.DataFrame:
    columns = parameterDF.columns
    lhReacs = [col for col in columns if " + LH ->" in col]

    nrows = len(parameterDF.index)
    rateConstants = np.empty(shape=(nrows, len(lhReacs)))
    for i, reaction in enumerate(lhReacs):
        rateConstants[:, i] = calculateRateConstantAllRows(
            parameterDF,
            reaction,
            Tdust,
            barrierWidth=barrierWidth,
            only_competition_fraction=only_competition_fraction,
        )

    return pd.DataFrame(rateConstants, columns=lhReacs)


def _get_UCLCHEM_dir() -> str:
    """Get the root directory of the UCLCHEM package

    Returns:
        uclchem_dir: directory of UCLCHEM
    """
    uclchem_dir = uclchem.__file__.split(os.path.sep)
    src_index = uclchem_dir.index("src")
    uclchem_dir = os.path.sep + os.path.join(*uclchem_dir[:src_index])
    return uclchem_dir


def finite_difference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    slope = np.diff(y) / np.diff(x)
    midpoints = np.diff(x) / 2 + x[:-1]
    return midpoints, slope


def convert_per_year_to_per_s(frequency: float | list[float]) -> float | list[float]:
    """Convert a frequency from per year to per second

    Inputs:
        frequency (float): frequency in per year

    Returns:
        float: frequency in per second
    """
    return frequency / (60.0 * 60.0 * 24.0 * 365.25)


# def _get_species_odes(param_dict, input_abundances):
#     input_abund = np.zeros(n_species)
#     input_abund[: len(input_abundances)] = input_abundances
#     # rate_indxs = np.ones(n_species)
#     # rate_indxs[: len(reac_indxs)] = reac_indxs
#     rates = wrap.get_odes(param_dict, input_abund)
#     # if success_flag < 0:
#     #    raise RuntimeError("UCLCHEM failed to return rates for these parameters")
#     return rates
#
#
# def get_rates_from_odes(species_name, df):
#     # Change this directory accordingly
#     _ROOT = _get_UCLCHEM_dir()
#     species = np.loadtxt(
#         os.path.join(_ROOT, "src", "uclchem", "species.csv"),
#         usecols=[0],
#         dtype=str,
#         skiprows=1,
#         unpack=True,
#         delimiter=",",
#         comments="%",
#     )
#     species = list(species)
#
#     species_index = species.index(species_name) + 1  # fortran index of species
#
#     rates_species = []
#     for i, row in df.iterrows():
#         # recreate the parameter dictionary needed to get accurate rates
#         param_dict = _param_dict_from_output(row)
#
#         # get the rate of all reactions from UCLCHEM
#         rates = _get_species_odes(param_dict, row[species])
#         rates_species.append(rates[species_index - 1])
#     return df["Time"], np.array(rates_species)


# def my_legend(axis=None):
#     if axis == None:
#         axis = plt.gca()
#
#     N = 32
#     lines = axis.lines
#     Nlines = len(lines)
#
#     xmin, xmax = axis.get_xlim()
#     ymin, ymax = axis.get_ylim()
#
#     # the 'point of presence' matrix
#     pop = np.zeros((Nlines, N, N), dtype=np.float64)
#
#     xscale = axis.get_xscale()
#     yscale = axis.get_yscale()
#     if xscale == "log":
#         xmin, xmax = np.log10(xmin), np.log10(xmax)
#     if yscale == "log":
#         ymin, ymax = np.log10(ymin), np.log10(ymax)
#     for l in range(Nlines):
#         # get xy data and scale it to the NxN squares
#         xy = lines[l].get_xydata()
#         if xscale == "log":
#             xy[:, 0] = np.log10(xy[:, 0])
#         if yscale == "log":
#             xy[:, 1] = np.log10(xy[:, 1])
#         xy = (xy - [xmin, ymin]) / ([xmax - xmin, ymax - ymin]) * N
#         xy = xy.astype(np.int32)
#         # mask stuff outside plot
#         mask = (xy[:, 0] >= 0) & (xy[:, 0] < N) & (xy[:, 1] >= 0) & (xy[:, 1] < N)
#         xy = xy[mask]
#         # add to pop
#         for p in xy:
#             pop[l][tuple(p)] = 1.0
#         pop[l][:, 0] = 0
#         pop[l][:, N - 1] = 0
#         pop[l][0, :] = 0
#         pop[l][N - 1, :] = 0
#
#     # find whitespace, nice place for labels
#     ws = 1.0 - (np.sum(pop, axis=0) > 0) * 1.0
#     # don't use the borders
#     ws[:, 0] = 0
#     ws[:, N - 1] = 0
#     ws[0, :] = 0
#     ws[N - 1, :] = 0
#
#     # blur the pop's
#     for l in range(Nlines):
#         pop[l] = ndimage.gaussian_filter(pop[l], sigma=N / 5)
#
#     for l in range(Nlines):
#         if lines[l].get_label()[0] == "_":
#             continue
#
#         # positive weights for current line, negative weight for others....
#         w = -0.3 * np.ones(Nlines, dtype=np.float64)
#         w[l] = 0.5
#
#         # calculate a field
#         p = ws + np.sum(w[:, np.newaxis, np.newaxis] * pop, axis=0)
#         plt.figure()
#         plt.imshow(p, interpolation="nearest")
#         plt.title(lines[l].get_label())
#
#         pos = np.argmax(p)  # note, argmax flattens the array first
#         best_x, best_y = (pos / N, pos % N)
#         x = xmin + (xmax - xmin) * best_x / N
#         y = ymin + (ymax - ymin) * best_y / N
#
#         if xscale == "log":
#             x = np.power(10, x)
#         if yscale == "log":
#             y = np.power(10, y)
#         axis.text(
#             x,
#             y,
#             lines[l].get_label(),
#             c=lines[l].get_color(),
#             horizontalalignment="center",
#             verticalalignment="center",
#         )


def rankData(X: np.ndarray) -> np.ndarray:
    """Rank data, with lowest value 1

    Inputs:
        X (np.ndarray): array to have its values ranked.
            If two-dimensional

    Returns:
        Xranks (np.ndarray):
    """
    if not isinstance(X, np.ndarray):
        msg = f"X should be of type np.ndarray, but was of type {type(X)}"
        raise TypeError(msg)

    if X.ndim > 2:
        msg = f"Array X had dimensionality {x.ndim}, but rankData is only implemented for 1- or 2-dimensional arrays."
        raise NotImplementedError(msg)
    elif X.ndim == 2:
        # 2-dimensional array, treated as array of arrays, rank values within each column
        Xranks = stats.rankdata(X, axis=0)
    else:
        Xranks = stats.rankdata(X)
    return Xranks


def fileIsNewerThanPythonChanges(pythonFilepath, compareFilepath):
    if not os.path.isfile(compareFilepath):
        return False
    if os.path.getmtime(pythonFilepath) <= os.path.getmtime(compareFilepath):
        return True
    else:
        return False


def is_dark_background() -> bool:
    return mpl.rcParams["axes.edgecolor"] == "#FFFFFF"


def rankInverseNormalTransform(X: np.ndarray, method: str = "bliss") -> np.ndarray:
    """Perform rank-based inverse normal transform on X

    Inputs:
        X (np.ndarray):
        method (str):

    Returns:

    """
    # RIN best for calculating CI of non-normally distributed data:
    #   https://sci-hub.se/https://doi.org/10.1037/a0028087
    #   https://doi.org/10.3758/s13428-016-0702-8

    # Different rank-based inverse normal transformations:
    #   https://pmc.ncbi.nlm.nih.gov/articles/PMC2921808/#abstract1

    method = method.lower()

    if method != "bliss":
        msg = (
            f"Only method 'bliss' is implemented at the moment, but method was {method}"
        )
        raise NotImplementedError(msg)

    Xranks = rankData(X)

    if method == "bliss":
        c = 0.5

    n = np.shape(X)[0]
    p = (Xranks - c) / (n - 2 * c + 1)

    return stats.norm.ppf(p)


def find_nearest_idx_sorted(
    array: np.ndarray, value: float | list[float]
) -> int | list[int]:
    """Find the index of a sorted array with value closest to value"""
    idx = np.searchsorted(array, value)
    if isinstance(value, list) or isinstance(value, np.ndarray):
        replaceByMinusOne = np.where(
            (idx > 0)
            & (
                (idx == len(array))
                | (np.abs(value - array[idx - 1]) < np.abs(value - array[idx]))
            )
        )
        idx[replaceByMinusOne] -= 1
        return idx
    else:
        if idx > 0 and (
            idx == len(array)
            or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
        ):
            return idx - 1
        else:
            return idx


def getTimeIndices(dfs: list[pd.DataFrame], time: float | list[float]) -> list[int]:
    """Get the row index where df["Time"] is closest to desired time for every df in dfs"""
    timeIndices = np.array(
        [find_nearest_idx_sorted(df["Time"].to_numpy(), time) for df in dfs]
    )
    return timeIndices


def getTimeFromIndices(
    dfs: list[pd.DataFrame], timeIndices: float | list[float]
) -> np.ndarray[float]:
    """Get the values of time from the array or matrix of timeIndices from every dataframe"""
    if timeIndices.ndim == 2:
        return np.array(
            [df["Time"].iloc[timeIndices[i, :]] for i, df in enumerate(dfs)]
        )
    elif timeIndicies.ndim == 1:
        return np.array([df["Time"].iloc[timeIndices[i]] for i, df in enumerate(dfs)])
    else:
        raise NotImplementedError()


def checkModelReachesDesiredTime(df: pd.DataFrame, maxDesiredTime: float) -> bool:
    """Checks that the model did reach the desired time. If it did not, it was terminated
    early, perhaps because it encountered some integration troubles.

    Inputs:
        df (pd.DataFrame): dataframe of model run containing column 'Time'
        maxDesiredTime (float): desired ending time of model

    Returns:
        bool: whether the model in df exceeded the desiredTime
    """
    return df["Time"].iloc[-1] >= maxDesiredTime


def removeModelsThatDontReachDesiredTime(
    dfs: list[pd.DataFrame], maxDesiredTime: float
) -> tuple[list[pd.DataFrame], list[int]]:
    # Get array. Values are True if the model did not finish, False if they did
    modelDidNotFinish = np.array(
        [~checkModelReachesDesiredTime(df, maxDesiredTime) for df in dfs]
    )

    # Count how many are True (i.e. they did not finish)
    count = np.count_nonzero(modelDidNotFinish)

    if count > 0:
        # If count > 0, not all models finished.
        print(
            f"Found {count} models that did not finish to the desired time {maxDesiredTime}. Filtering"
        )
        dfs = [dfs[i] for i in range(len(dfs)) if not modelDidNotFinish[i]]
        indicesToKeep = np.where(~modelDidNotFinish)[0]
    else:
        # If count == 0, all models finished, we can just keep all indices
        indicesToKeep = np.arange(len(dfs))
    return dfs, indicesToKeep


def checkTimeIndices(dfs, times, timeIndices, desiredTimes, quiet=True):
    # Check whether the timeIndices actually correspond to times close to the desiredTimes
    if not quiet:
        print(f"Checking desired times and the corresponding time indices:")
    timeDifference = np.abs(np.subtract(times, desiredTimes))
    maxTimeDifference = round(np.max(timeDifference), 2)
    maxIndex = np.argmax(timeDifference)
    maxIndex = np.unravel_index(maxIndex, np.shape(timeDifference))
    msg = f"""\tMaximum deviation from desired time: {maxTimeDifference} years
    \tOccurs in sample {maxIndex[0]} at desiredTime {round(desiredTimes[maxIndex[1]], 2)} years
    \tAt small desiredTimes (<1e4 years), the timestep is not yet tight, so if that is the case it's no cause for concern.
    \tSample {maxIndex[0]} has maximum time {round(dfs[maxIndex[0]]["Time"].iloc[-1], 2)} years.
    \tIf this maximum time is much smaller than the desired time, then the model did not finish correctly.
    \tIf so, maybe remove this sample from the future calculations?"""
    if not quiet:
        print(msg)

    if not quiet:
        print("Checking whether there are any duplicate values:")
    timesDiff = np.diff(times, axis=1, prepend=0)
    timesDiffIsZero = np.all(np.isclose(timesDiff, 0), axis=0)
    count = np.count_nonzero(timesDiffIsZero)
    if not quiet:
        print(f"\tFound {count} duplicate time values")
    if count > 0:
        if not quiet:
            print(f"\tRemoving {count} duplicate indices")
        reducedTimeIndices = timeIndices[:, ~timesDiffIsZero]
        reducedTimes = times[:, ~timesDiffIsZero]
    else:
        if not quiet:
            print(f"\tSimply returning timeIndices and time changed")
        reducedTimeIndices = timeIndices
        reducedTimes = times
    return reducedTimeIndices, reducedTimes


def getAbundance(
    df: pd.DataFrame,
    timeIndex: int | list[int],
    spec: str,
    on_grain: bool,
    only_surf: bool = False,
    only_bulk: bool = False,
) -> float:
    """Get the abundance of a certain species in a certain model run at a certain row indicated by timeIndex,
    or set of rows if timeIndex is a list or np array

    Inputs:
        df (pd.DataFrame): dataframe of model run
        timeIndex (int | list[int]): integer or list of integers of rows to use
        spec (str): species string. Does not include '#' or '@'
        on_grain (bool): whether to get the grain abundance
        only_surf (bool): whether to use only the surface abundance
        only_bulk (bool): whether to use only the bulk abundance

    Returns:
        float | np.ndarray: abundance or abundances with respect to hydrogen nuclei
    """
    if "#" in spec or "@" in spec:
        mgs = "'#' or '@' found in spec argument. If you desire only to get surface or only bulk arguments, please use only_surf=True or only_bulk=True accordingly (with on_grain=True also)"
        raise ValueError(msg)
    if (only_surf and not on_grain) or (only_bulk and not on_grain):
        msg = "Only_surf or only_bulk was True, but on_grain was not. Please use on_grain=True also"
        raise ValueError(msg)
    if only_surf and only_bulk:
        msg = "Both only_surf and only_bulk was True, but only one can be true"
        raise ValueError(msg)
    if on_grain:
        if only_surf:
            return df["#" + spec].iloc[timeIndex]
        else:
            return df["#" + spec].iloc[timeIndex] + df["@" + spec].iloc[timeIndex]
    else:
        return df[spec].iloc[timeIndex]


def getAbundances(
    dfs: list[pd.DataFrame],
    timeIndices: list[int],
    spec: str,
    on_grain: bool = False,
    only_surf: bool = False,
    only_bulk: bool = False,
    njobs: int = 1,
) -> np.ndarray:
    """Get the abundance of a certain species in a list of model runs, either at one or multiple timepoints.

    Inputs:
        dfs (list): list of dataframes of model runs
        timeIndices (list[int]): list of integers of row to use in each df, corresponding to the same time
        spec (str): species string. Does not include '#' or '@'
        on_grain (bool): whether to get the grain abundance
        only_surf (bool): whether to use only the surface abundance
        only_bulk (bool): whether to use only the bulk abundance
        njobs (int): number of threads to use. Each thread reads one dataframe

    Returns:
        float | np.ndarray: abundance or abundances with respect to hydrogen nuclei
    """
    if timeIndices.ndim == 2:
        if njobs == 1:
            abundances = [
                getAbundance(
                    df,
                    timeIndices[i, :],
                    spec,
                    on_grain,
                    only_surf=only_surf,
                    only_bulk=only_bulk,
                )
                for i, df in enumerate(dfs)
            ]
        else:
            args = [
                (df, timeIndices[i, :], spec, on_grain, only_surf, only_bulk)
                for i, df in enumerate(dfs)
            ]
            with Pool(processes=min(njobs, cpu_count())) as pool:
                abundances = pool.starmap(getAbundance, args)
    else:
        if njobs == 1:
            abundances = [
                getAbundance(
                    df,
                    timeIndices[i],
                    spec,
                    on_grain,
                    only_surf=only_surf,
                    only_bulk=only_bulk,
                )
                for i, df in enumerate(dfs)
            ]
        else:
            args = [
                (df, timeIndices[i], spec, on_grain, only_surf, only_bulk)
                for i, df in enumerate(dfs)
            ]
            with Pool(processes=min(njobs, cpu_count())) as pool:
                abundances = pool.starmap(getAbundance, args)
    return np.array(abundances)


def getConfidenceIntervalsOfAbundances(
    abundances: np.ndarray, confidence_level: float = 0.6827
) -> tuple[np.ndarray, np.ndarray]:
    """Get confidence level of abundances

    Inputs:
        abundances (np.ndarray): abundances
        confidence_level (float): desired confidence level, between 0 and 1.

    Returns:
        cilow (np.ndarray): low bound of confidence interval
        cihigh (np.ndarray): high bound of confidence interval
    """
    cilow = np.quantile(abundances, (1.0 - confidence_level) / 2.0, axis=0)
    cihigh = np.quantile(abundances, (1.0 + confidence_level) / 2.0, axis=0)
    return cilow, cihigh


def getLogMeanOfAbundances(abundances: np.ndarray) -> np.ndarray:
    """Get the mean of the logarithm of the abundances

    Inputs:
        abundnaces (np.ndarray): abundances in all samples.

    Returns:
        np.ndarray: log-mean of abundances
    """
    return np.exp(np.mean(np.log(abundances), axis=0))


def is_significant(res, minStatistic: float = 0.4, maxPvalue: float = 0.05) -> bool:
    """Checks whether a res is significant, i.e. has res.statistic >= minStatistic and res.pvalue <= maxPvalue

    Inputs:
        res: result from for example scipy.stats.pearsonr
        minStatistic (float): minimum value of statistic
        maxPvalue (float): maximum pvalue of res for it to be considered significant

    Returns:
        bool: whether a res is significant
    """
    checkStatistic(minStatistic)
    checkPvalue(maxPvalue)
    if not hasattr(res, "statistic") or not hasattr(res, "pvalue"):
        msg = "res should have attributes 'statistic' and 'pvalue', but did not."
        raise AttributeError(msg)
    return math.fabs(res.statistic) >= minStatistic and res.pvalue <= maxPvalue


def getPhysicalParameters(filepath: str) -> list[float]:
    """Extract the physical parameters of a model run from the filepath"""
    physicalParams = (
        os.path.splitext(filepath)[0].split(os.path.sep)[-1].split("_")[:-1]
    )
    return [float(i) for i in physicalParams]


def getRunNr(filepath: str) -> int:
    """Extract the run number of a model run from the filepath"""
    return int(os.path.splitext(filepath)[0].split("_")[-1].strip("sample"))


def getPhysicalParamSets(filepaths: list[str]) -> list[list[float]]:
    """Get all the physical parameter sets in the filepaths"""
    physicalParamSets = list(
        physParam
        for physParam, _ in itertools.groupby(
            [getPhysicalParameters(i) for i in filepaths]
        )
    )
    seenParamSets = []
    for paramSet in physicalParamSets:
        if paramSet in seenParamSets:
            continue
        seenParamSets.append(paramSet)
    seenParamSets.sort()
    return seenParamSets


def sortByPhysicalParamSet(filepaths: list[str]):
    """Sort a list of filepaths by the physical parameters within the filepath."""
    physicalParamSets = getPhysicalParamSets(filepaths)
    filesForParamSets = [[]] * len(physicalParamSets)
    for i, paramSet in enumerate(physicalParamSets):
        string = physicalParamSetToPathString(paramSet)
        filesForParamSet = [file for file in filepaths if string in file]
        if len(filesForParamSet) > 1:
            filesForParamSet.sort(key=getRunNr)
        filesForParamSets[i] = filesForParamSet
    return filesForParamSets


def physicalParamSetToPathString(physicalParams: list[float]) -> str:
    """Convert a list of physical parameters to a string as they would be in a model run path by joining them with '_'"""
    return "_".join(str(i) for i in physicalParams)


def physicalParamSetToString(physicalParams: list[float]) -> str:
    """Convert a list of physical parameters to a nice string for human-readable, with units and symbols"""
    return f"T = {physicalParams[0]} K, nH = {physicalParams[1]:.1e} cm-3, zeta = {physicalParams[2] * 1.3e-17} s-1, UV = {physicalParams[3]} Habing"


def physicalParamSetToSaveString(physicalParams: list[float]) -> str:
    """Convert a list of physical parameters to a nice string for human-readable, without symbols but with units"""
    return f"{physicalParams[0]}K_{physicalParams[1]:.1e}cm-3_{physicalParams[2] * 1.3e-17}_{physicalParams[3]}Habing"


def physicalParamSetToIndex(
    physicalParameterSets: list[list[float]],
    T: float = None,
    nH: float = None,
    zeta: float = None,
    radfield: float = None,
) -> int:
    if T is None or nH is None or zeta is None or radfield is None:
        raise ValueError()
    return physicalParameterSets.index([T, nH, zeta, radfield])


def getDataFramesForPhysicalParamSet(
    physicalParamIndex: int,
    filepathsSamples: list[list[str]],
    filepathsNominal: list[list[str]],
    njobs: int = 1,
    format: str | None = None,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    samplesAtPhysicalParam = filepathsSamples[physicalParamIndex]
    nominalAtPhysicalParam = filepathsNominal[physicalParamIndex][0]

    sampleDFs = readOutputFiles(samplesAtPhysicalParam, njobs=njobs, format=format)
    if format is None:
        format = nominalAtPhysicalParam.split(os.path.extsep)[-1]
        print(f"Output file format of nominal df was detected to be: {format}")
    elif not isinstance(format, str):
        msg = f"Format should be a string (one of ['csv', 'h5', 'hdf5']) or None, but it was type {type(format)}"
        raise TypeError(msg)
    format = format.lower()
    if format not in ["csv", "h5", "hdf5"]:
        msg = f"format should be one of ['csv', 'h5', 'hdf5'], but it was {format}"
        raise ValueError(msg)
    if format == "csv":
        nominalDF = read_output_file(nominalAtPhysicalParam)
    else:
        nominalDF = read_output_file_h5(nominalAtPhysicalParam)
    return sampleDFs, nominalDF


def calculateCorrelation(
    x: list[float],
    y: list[float],
    confidence_level: float = 0.95,
    bootstrapmethod=stats.BootstrapMethod(n_resamples=2000, batch=500, method="BCa"),
    calculateConfidenceInterval: bool = True,
) -> tuple[float, float, float, float]:
    """Calculate the pearson correlation coefficient of x and y, and if desired,
    also calculate its confidence interval using a bootstrap method."""
    if len(x) != len(y):
        raise ValueError()

    res = stats.pearsonr(x, y, alternative="two-sided")
    if calculateConfidenceInterval:
        CIlow, CIhigh = res.confidence_interval(
            confidence_level=confidence_level, method=bootstrapmethod
        )
    else:
        CIlow, CIhigh = None, None
    return res.statistic, res.pvalue, CIlow, CIhigh


def calculateSignificantCorrelations2D(
    abundancesRIN,
    parametersRIN,
    parameterNames: list[str],
    confidence_level: float = 0.95,
    minStatistic: float = 0.5,
    maxPvalue: float = 0.05,
):
    correlations = calculateAllCorrelations2D(
        abundancesRIN,
        parametersRIN,
        parameterNames,
        calculateConfidenceInterval=False,
    )

    # Find which parameter indices are at any timepoint strong enough and significant
    sigIndices = getSignificantCorrelationsIndices(
        correlations, minStatistic=minStatistic, maxPvalue=maxPvalue
    )

    if not sigIndices:
        # If there are none, continue to the next species
        return None

    # Recalculate the strong and significant correlations, now also calculate their confidence intervals
    correlations = calculateAllCorrelations2D(
        abundancesRIN,
        parametersRIN[:, sigIndices],
        parameterNames[sigIndices],
        calculateConfidenceInterval=True,
        confidence_level=0.95,
    )
    return correlations


def calculateAllCorrelations2D(
    abundancesRIN: np.ndarray,
    parametersRIN: np.ndarray,
    parameterNames: list[str],
    confidence_level: float = 0.95,
    calculateConfidenceInterval=False,
) -> pd.DataFrame:
    """Calculate the correlation coefficients of all parameters for a 2D array of abundancesRIN"""

    nparams = len(parameterNames)
    nabunds = np.shape(abundancesRIN)[1]

    if not nparams == np.shape(parametersRIN)[1]:
        msg = f"Number of parameter names in parameterNames ({nparams}) is not the same as the number of samples in abundancesRIN array ({np.shape(parametersRIN)[0]})"
        raise ValueError(msg)

    if not np.shape(parametersRIN)[0] == np.shape(abundancesRIN)[0]:
        msg = f"Number of samples in parametersRIN array ({np.shape(parametersRIN)[0]}) is not the same as the number of samples in abundancesRIN array ({np.shape(abundancesRIN)[0]})"
        raise ValueError(msg)

    # Here, we want to calculate all pearson coeffs for each parameter at each different column of abundancesRIN,
    # such that we get an N*M result, where N is the number of parameters and M the number of columns in abundancesRIN
    res = stats.pearsonr(
        parametersRIN[:, :, np.newaxis],
        abundancesRIN[:, np.newaxis, :],
        axis=0,
        alternative="two-sided",
    )

    if calculateConfidenceInterval:
        bootstrapmethod = stats.BootstrapMethod(
            n_resamples=2000, batch=500, method="BCa"
        )
        CIlow, CIhigh = res.confidence_interval(
            confidence_level=confidence_level, method=bootstrapmethod
        )
        CIDiffDown, CIDiffUp = res.statistic - CIlow, CIhigh - res.statistic
        # plt.figure()
        # plt.scatter(res.statistic.flatten(), CIDiffDown.flatten(), label="Up")
        # plt.scatter(res.statistic.flatten(), CIDiffUp.flatten(), label="Down")
        # plt.axhline(0, c="gray", ls="dashed")
        # plt.axvline(0, c="gray", ls="dashed")
        # plt.xlim([-1, 1])
        # plt.ylabel("Confidence interval")
        # plt.ylabel("Statistic")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
    else:
        CIlow, CIhigh = (
            np.full(shape=(nparams, nabunds), fill_value=None),
            np.full(shape=(nparams, nabunds), fill_value=None),
        )
        CIDiffDown, CIDiffUp = (
            np.full(shape=(nparams, nabunds), fill_value=None),
            np.full(shape=(nparams, nabunds), fill_value=None),
        )

    valuesDF = pd.DataFrame(
        np.array(
            [
                parameterNames,
                res.statistic.tolist(),
                res.pvalue.tolist(),
                CIlow.tolist(),
                CIhigh.tolist(),
                CIDiffDown.tolist(),
                CIDiffUp.tolist(),
            ],
            dtype=object,
        ).T,
        columns=[
            "parameter",
            "statistic",
            "pvalue",
            "cilow",
            "cihigh",
            "cidiffdown",
            "cidiffup",
        ],
    )
    return valuesDF


def calculateAllCorrelations(
    abundancesRIN: list[float],
    parametersRIN: np.ndarray,
    parameterNames: list[str],
    confidence_level: float = 0.95,
    calculateConfidenceInterval=False,
) -> pd.DataFrame:
    """Calculate the correlation coefficients of all parameters"""

    if abundancesRIN.ndim == 2:
        return calculateAllCorrelations2D(
            abundancesRIN,
            parametersRIN,
            parameterNames,
            confidence_level,
            calculateConfidenceInterval,
        )

    nparams = len(parameterNames)

    if not nparams == np.shape(parametersRIN)[1]:
        msg = f"Number of parameter names in parameterNames ({nparams}) is not the same as the number of samples in abundancesRIN array ({np.shape(parametersRIN)[0]})"
        raise ValueError(msg)

    if not np.shape(parametersRIN)[0] == np.shape(abundancesRIN)[0]:
        msg = f"Number of samples in parametersRIN array ({np.shape(parametersRIN)[0]}) is not the same as the number of samples in abundancesRIN array ({np.shape(abundancesRIN)[0]})"
        raise ValueError(msg)

    res = stats.pearsonr(
        parametersRIN.T, abundancesRIN, axis=-1, alternative="two-sided"
    )
    if calculateConfidenceInterval:
        bootstrapmethod = stats.BootstrapMethod(
            n_resamples=2000, batch=1000, method="BCa"
        )
        CIlow, CIhigh = res.confidence_interval(
            confidence_level=confidence_level, method=bootstrapmethod
        )
        CIDiffDown, CIDiffUp = res.statistic - CIlow, CIhigh - res.statistic
    else:
        CIlow, CIhigh = [None] * nparams, [None] * nparams
        CIDiffDown, CIDiffUp = [None] * nparams, [None] * nparams

    valuesDF = pd.DataFrame(
        np.array(
            [
                parameterNames,
                res.statistic,
                res.pvalue,
                CIlow,
                CIhigh,
                CIDiffDown,
                CIDiffUp,
            ]
        ).T,
        columns=[
            "parameter",
            "statistic",
            "pvalue",
            "cilow",
            "cihigh",
            "cidiffdown",
            "cidiffup",
        ],
    )
    return valuesDF


def findAllNumbers(string: str) -> dict:
    """Find all numbers in a string.

    Inputs:
        string (str):

    Returns:
        dct (dict):
    """
    if not isinstance(string, str):
        raise TypeError()
    dct = dict((m.start(), m.group()) for m in re.finditer(r"\d+", string))
    return dct


def checkStatistic(statistic: float) -> None:
    """Checks that the statistic is between -1 and 1.

    Inputs:
        statistic (float): statistic
    """
    if not isinstance(statistic, float):
        msg = f"statistic should be type float, but was type {type(statistic)}"
        raise TypeError(msg)
    if abs(statistic) > 1.0:
        msg = f"statistic should be between -1 and 1, but was {statistic}"
        raise ValueError(msg)


def checkPvalue(pvalue: float) -> None:
    if not isinstance(pvalue, float):
        msg = f"pvalue should be type float, but was type {type(pvalue)}"
        raise TypeError()
    if pvalue < 0.0 or pvalue > 1.0:
        msg = f"pvalue should be between 0 and 1, but was {pvalue}"
        raise ValueError()


def convertDensityToTitle(density):
    log10Density = int(np.log10(density))
    return f"$10^{{{log10Density}}}$ cm$^{{-3}}$"


def convertTimeToTitle(time):
    log10Time = int(np.log10(time))
    remainder = time / (10**log10Time)
    if int(remainder) != remainder:
        return f"${{{remainder}}}\\times 10^{{{log10Time}}}$ years"
    else:
        return f"$10^{{{log10Time}}}$ years"


def convertSpeciesToLegendLabel(
    species: str | list[str],
) -> str | list[str]:
    """Convert a species name or list of species names to proper legend labels with subscripts where necessary"""
    if isinstance(species, list):
        return [convertSpeciesToLegendLabel(spec) for spec in species]
    # Indication that something is surface is not necessary, as everything is surface
    species = species.replace("#", "")
    # Element strings all capitalized in UCLCHEM output
    species = species.replace("SI", "Si")

    numbers = findAllNumbers(species)
    to_skip = 0
    for j, number in numbers.items():
        species = (
            species[: j + to_skip]
            + r"$_{{{0}}}$".format(number)
            + species[j + to_skip + len(number) :]
        )
        to_skip += len(number) + 4
    return species


def convertParameterNameToLegendLabel(param: str) -> str:
    """Convert the parameter name to a proper legend label"""
    if False:
        # Indication that something is surface is not necessary, as everything is surface
        param = param.replace("#", "")
        # Attempt to work with chemformula package, did not work.
        # No arrow drawn.
        if "LH" in param:
            param = param.replace(" + LH", "")
            print(param)
            return r"\ch{" + param + "}"
        elif "prefac" in param:
            splitParam = param.split()
            if splitParam[1] == "desprefac":
                splitParam[1] = r"$\nu_{\mathrm{des}}$"
            elif splitParam[1] == "diffprefac":
                splitParam[1] = r"$\nu_{\mathrm{diff}}$"
            else:
                raise NotImplementedError()
            return " ".join([r"\ch{" + splitParam[0] + "}", splitParam[1]])
        else:
            splitParam = param.split()
            if splitParam[1] == "bind":
                splitParam[1] = r"$E_{\mathrm{bind}}$"
            elif splitParam[1] == "diff":
                splitParam[1] = r"$E_{\mathrm{diff}}$"
            else:
                raise NotImplementedError()
            return " ".join([r"\ch{" + splitParam[0] + "}", splitParam[1]])
    else:
        # Convert species name(s) to have correct subscripts
        param = convertSpeciesToLegendLabel(param)
        if "prefac" in param:
            if "des" in param:
                param = param.replace("desprefac", r"$\nu_{\mathrm{des}}$")
            elif "diff" in param:
                param = param.replace("diffprefac", r"$\nu_{\mathrm{diff}}$")
            else:
                # Not implemented type of prefactor
                raise NotImplementedError()
        elif "->" not in param:
            if "bind" in param:
                param = param.replace("bind", r"$E_{\mathrm{bind}}$")
            elif "diff" in param:
                param = param.replace("diff", r"$E_{\mathrm{diff}}$")
            else:
                raise NotImplementedError()
        elif "LH" in param:
            param = param.replace("+ LH ->", r"$\longrightarrow$")
        else:
            raise NotImplementedError()
        return param


def convertParameterNameToAxisLabel(param: str) -> str:
    """Convert the parameter name to a proper axis label, including unit"""
    param = convertParameterNameToLegendLabel(param)
    if "nu" in param:
        param += " (s$^{-1}$)"
    elif "$E" in param or r"$\rightarrow$" in param:
        param += " (K)"
    return param


def getColors() -> list[str]:
    """Get all colors in the current matplotlib color cycler"""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors


def getAllRunsFilepaths(
    sampleRunsDir: str, extension="dat"
) -> tuple[list[list[str]], list[list[str]]]:
    """Get all model runs in a certain directory, split them by whether they're nomimal or not,
    and sort them according to the physical parameter set and run number."""
    if extension[0] == ".":
        print(
            f"'.' was found at the start of extension {extension}, but is added by function getAllRunsFilepaths. Removing it."
        )
        extension = extension[1:]

    matchStringSamples = os.path.join(sampleRunsDir, f"*_sample*.{extension}")
    matchStringNominal = os.path.join(sampleRunsDir, f"*_sampleNominal.{extension}")

    filepathsSample = glob(matchStringSamples)
    filepathsSample = [
        filepath for filepath in filepathsSample if not "Nominal" in filepath
    ]
    filepathsNominal = glob(matchStringNominal)

    if not filepathsSample or not filepathsNominal:
        raise ValueError()

    return sortByPhysicalParamSet(filepathsSample), sortByPhysicalParamSet(
        filepathsNominal
    )


# def integrateAbsoluteCorrelations(
#     correlationsDF, timeArray, timescale: str | None = None
# ):
#     statistic = np.array(correlationsDF["statistic"].to_list())
#     statistic = np.abs(statistic)
#     if timeArray.ndim == 2:
#         timeArray = timeArray[0, :]
#     if timescale is not None:
#         timescale = timescale.lower()
#         if timescale == "none":
#             pass
#         elif timescale == "log":
#             timeArray = np.log10(timeArray)
#         elif timescale == "neglect":
#             timeArray = np.arange(np.shape(statistic)[1])
#         else:
#             raise ValueError()
#
#     integrals = integrate.trapezoid(statistic, timeArray, axis=1)
#     return integrals
#
#
# def addIntegralColumnToCorrelationsDF(
#     correlationsDF, timeArray, timescale: str | None = None
# ):
#     integrals = integrateAbsoluteCorrelations(correlationsDF, timeArray)
#     correlationsDF["integrals"] = integrals


def read_output_file_h5(filepath: str) -> pd.DataFrame:
    data = pd.read_hdf(filepath)
    data.columns = data.columns.str.strip()
    return data


def readOutputFiles(
    filepaths: list[str], njobs=1, format: str | None = None
) -> list[pd.DataFrame]:
    if format is None:
        format = filepaths[0].split(os.path.extsep)[-1]
        print(f"Output file format was detected to be: {format}")
    elif not isinstance(format, str):
        msg = f"Format should be a string (one of ['csv', 'h5', 'hdf5']) or None, but it was type {type(format)}"
        raise TypeError(msg)
    format = format.lower()
    if format not in ["csv", "h5", "hdf5"]:
        msg = f"format should be one of ['csv', 'h5', 'hdf5'], but it was {format}"
        raise ValueError(msg)

    if format == "csv":
        if njobs == 1:
            dfs = [read_output_file(filepath) for filepath in filepaths]
        else:
            with Pool(processes=min(njobs, cpu_count())) as pool:
                dfs = pool.map(read_output_file, filepaths)
    else:
        # If format is not csv, it must be hdf5
        if njobs == 1:
            dfs = [read_output_file_h5(filepath) for filepath in filepaths]
        else:
            with Pool(processes=min(njobs, cpu_count())) as pool:
                dfs = pool.map(read_output_file_h5, filepaths)
    return dfs


def getSignificantCorrelationsIndices(
    correlations: pd.DataFrame, minStatistic: float = 0.5, maxPvalue: float = 0.05
) -> list[int]:
    """Get indices of all the rows where the statistic is larger than minStatistic
    and the pvalue smaller than maxPvalue"""
    checkStatistic(minStatistic)
    checkPvalue(maxPvalue)

    if not isinstance(correlations, pd.DataFrame):
        raise TypeError()
    if isinstance(correlations["statistic"].iloc[0], list):
        indices = np.where(
            [
                np.any(
                    (np.abs(row["statistic"]) >= minStatistic)
                    & (np.asarray(row["pvalue"]) <= maxPvalue)
                )
                for i, row in correlations.iterrows()
            ]
        )[0]
    else:
        indices = np.where(
            (correlations["statistic"].abs() >= minStatistic)
            & (correlations["pvalue"] <= maxPvalue)
        )[0]
    return list(indices)


def getTotalRuntime(allFiles: list[str]) -> float:
    """Get the time it took from the creation of the first file to the most recent change of any file in seconds"""
    # Low-zeta:
    lowZetaFiles = [file for file in allFiles if not "_100.0_" in file]
    mostRecentChange = np.max([os.path.getctime(filepath) for filepath in lowZetaFiles])
    oldestCreation = np.min([os.path.getmtime(filepath) for filepath in lowZetaFiles])

    time = mostRecentChange - oldestCreation

    # High-zeta
    highZetaFiles = [file for file in allFiles if "_100.0_" in file]
    mostRecentChange = np.max([os.path.getctime(filepath) for filepath in lowZetaFiles])
    oldestCreation = np.min([os.path.getmtime(filepath) for filepath in lowZetaFiles])

    time += mostRecentChange - oldestCreation

    return time


def plotDataFrameAbundances(
    dfs,
    spec,
    on_grain: bool = False,
    ax=None,
    **kwargs,
) -> None:
    if ax is None:
        ax = plt.gca()
    if "#" in spec or "@" in spec:
        raise ValueError()

    for j, df in enumerate(dfs):
        if on_grain:
            abund = df["#" + spec] + df["@" + spec]
        else:
            abund = df[spec]
        ax.plot(df["Time"], abund, c="k", lw=0.2, alpha=0.1, **kwargs)


def plotAbundanceAgainstParameter(
    abundances, spec, parameterName, parameterDF, axis=None, rank=False, **kwargs
):
    if axis is None:
        axis = plt.gca()
    x = parameterDF[parameterName].values
    y = abundances
    if rank:
        x = rankData(x)
        y = rankData(y)
    else:
        axis.set_yscale("log")
        if "prefac" in parameterName:
            axis.set_xscale("log")
    axis.scatter(x, y, **kwargs)
    axis.set_xlabel(convertParameterNameToAxisLabel(parameterName))
    axis.set_ylabel(convertSpeciesToLegendLabel(spec))


def format_axis_as_log(ax, marker_step: int = 1) -> None:
    ax.set_major_formatter(ticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    min, max = ax.get_data_interval()
    tick_range = np.arange(np.floor(min), max, marker_step)
    ax.set_ticks(tick_range)
    if marker_step == 1:
        ax.set_ticks(
            [
                np.log10(x)
                for p in tick_range
                for x in np.linspace(10**p, 10 ** (p + 1), 10)
            ],
            minor=True,
        )
    else:
        ax.minorticks_off()


def createViolinPlot(
    abundances,
    specs,
    axis=None,
    color: list[str] | str | None = None,
    marker_step: int = 1,
    **kwargs,
):
    if axis is None:
        axis = plt.gca()

    positions = range(len(specs))

    logabundances = np.log10(abundances)
    violin_parts = axis.violinplot(
        dataset=logabundances,
        points=500,
        showmeans=False,
        showextrema=True,
        positions=positions,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([convertSpeciesToLegendLabel(spec) for spec in specs])
    axis.xaxis.minorticks_off()

    axis.scatter(positions, np.mean(logabundances, axis=0), c=color)

    axis.set_ylabel("Abundances (wrt H)")

    format_axis_as_log(axis.yaxis, marker_step=marker_step)

    if color is None:
        return
    if isinstance(color, str):
        color = [color] * len(specs)

    for i in range(len(specs)):
        violin_parts["bodies"][i].set_color(color[i])
    for part in violin_parts.keys():
        if part == "cbars":
            violin_parts[part].set_visible(False)
        elif part.startswith("c"):
            violin_parts[part].set_color(color)


def createRidgePlot(
    abundances: np.ndarray,
    overlap: float = 0.5,
    npoints: int = 100,
    axis=None,
    marker_step: int = 1,
    fill: bool = False,
    xlim: list[float] | None = None,
    color=None,
    padding_x: float = 0.5,
    vertical: bool = False,
):
    if axis is None:
        axis = plt.gca()
    shape = np.shape(abundances)
    ndistributions = shape[1]

    logabundances = np.log10(abundances)

    if isinstance(color, LinearSegmentedColormap):
        color = color.resampled(ndistributions)
        color = [color(i) for i in range(ndistributions)]

    for i in range(ndistributions):
        kde = stats.gaussian_kde(logabundances[:, i])

        minlogabundance = np.min(logabundances[:, i])
        maxlogabundance = np.max(logabundances[:, i])
        minx_kde = minlogabundance - padding_x
        maxx_kde = maxlogabundance + padding_x
        x = np.linspace(minx_kde, maxx_kde, endpoint=True, num=npoints)
        kde_evaluated = kde.evaluate(x)

        # Scale kde to be between 0 and 1
        kde_scaled = (kde_evaluated - np.min(kde_evaluated)) / (
            np.max(kde_evaluated) - np.min(kde_evaluated)
        )

        y_baseline = i * (1.0 - overlap)

        kwargs = {}
        # Plot. zorder = -i such that top plots are "behind" bottom plots,
        # and all plots are below axis stuff.
        if fill:
            if vertical:
                axis.fill_betweenx(
                    x,
                    np.ones(npoints) * y_baseline,
                    kde_scaled + y_baseline,
                    zorder=-i,
                    color=color[i],
                )
            else:
                axis.fill_between(
                    x,
                    np.ones(npoints) * y_baseline,
                    kde_scaled + y_baseline,
                    zorder=-i,
                    color=color[i],
                )

            kwargs["c"] = "w"
            kwargs["lw"] = 1.0
        else:
            kwargs["c"] = color[i]
        if vertical:
            axis.plot(
                kde_scaled + y_baseline,
                x,
                zorder=-i,
                **kwargs,
            )
            axis.axvline(
                y_baseline, c="k", alpha=0.2, lw=0.1, zorder=-ndistributions - 1
            )
        else:
            axis.plot(x, kde_scaled + y_baseline, zorder=-i, **kwargs)
            axis.axhline(
                y_baseline, c="k", alpha=0.2, lw=0.1, zorder=-ndistributions - 1
            )

    if xlim is None:
        xmin, xmax = (
            np.min(logabundances) - padding_x,
            np.max(logabundances) + padding_x,
        )
    else:
        xmin, xmax = np.log10(xlim)

    if vertical:
        axis.set_ylabel("Abundances (wrt H)")
        format_axis_as_log(axis.yaxis, marker_step=marker_step)

        plt.ylim([xmin, xmax])

        axis.xaxis.set_tick_params(
            which="both", top=False, bottom=False, labelbottom=False
        )
    else:
        axis.set_xlabel("Abundances (wrt H)")
        format_axis_as_log(axis.xaxis, marker_step=marker_step)

        plt.xlim([xmin, xmax])

        axis.yaxis.set_tick_params(
            which="both", left=False, right=False, labelleft=False
        )


def test_convergence(
    sampleDFs: list[pd.DataFrame],
    specs: list[str],
    timeIndices: list[float],
    parameters: pd.DataFrame,
    parameterNames: list[str],
    on_grain: bool = False,
    njobs: int = 1,
):
    nsamples = len(sampleDFs)
    nspecs = len(specs)
    nparameters = len(parameterNames)
    abundances_for_each_spec = np.empty(shape=(nsamples, nspecs))
    for i, spec in enumerate(specs):
        abundances_for_each_spec[:, i] = getAbundances(
            sampleDFs, timeIndices, spec, on_grain=on_grain, njobs=njobs
        )

    indeces = list(range(nsamples))
    num_samples = np.logspace(
        1, np.log10(min(1500, nsamples)), endpoint=True, dtype=int
    )

    statistics_for_nums = np.empty(shape=(nspecs, len(num_samples), nparameters))

    for i, num in enumerate(num_samples):
        shuffle(indeces)
        indeces_to_use = np.array(indeces)[:num]

        abundances_to_use = abundances_for_each_spec[indeces_to_use, :]
        abundances_to_use_RIN = rankInverseNormalTransform(abundances_to_use)

        parameters_to_use = parameters[indeces_to_use, :]
        parameters_to_use_RIN = rankInverseNormalTransform(parameters_to_use)
        for j, spec in enumerate(specs):
            result_df = calculateAllCorrelations(
                abundances_to_use_RIN[:, j], parameters_to_use_RIN, parameterNames
            )

            statistics_for_nums[j, i, :] = result_df["statistic"].values

    fig, axs = plt.subplots(1, nspecs, sharex=True, sharey=True)
    for j, spec in enumerate(specs):
        axs[j].set_title(convertSpeciesToLegendLabel(spec))
        for k, parameterName in enumerate(parameterNames):
            axs[j].plot(num_samples, statistics_for_nums[j, :, k], c="k", alpha=0.1)
    axs[j].set_xscale("log")
    axs[0].set_ylabel("$r_{\mathrm{RIN}}\\left(N\\right)$")
    fig.supxlabel("N")
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(1, nspecs, sharex=True, sharey=True)
    for j, spec in enumerate(specs):
        axs[j].set_title(convertSpeciesToLegendLabel(spec))
        abs_diff = np.abs(statistics_for_nums[j, :, :] - statistics_for_nums[j, -1, :])
        for k, parameterName in enumerate(parameterNames):
            axs[j].plot(num_samples, abs_diff[:, k], c="k", alpha=0.1)
    axs[j].set_xscale("log")
    axs[0].set_ylabel(
        "$\\left|r_{\mathrm{RIN}}\\left(N\\right)-r_{\mathrm{RIN}}\\left(1000\\right)\\right|$"
    )
    axs[0].set_yscale("log")
    fig.supxlabel("$N$")
    plt.tight_layout()
    plt.show()


def compare_with_converged(
    sampleDFs: list[pd.DataFrame],
    convergenceDFs: list[pd.DataFrame],
    specs: list[str],
    time: float,
    sampleParameters: pd.DataFrame,
    convergenceParameters: pd.DataFrame,
    parameterNames: list[str],
    on_grain: bool = False,
    njobs: int = 1,
    savefig_path: str | None = None,
):
    nspecs = len(specs)

    nconvergence = len(convergenceDFs)
    timeIndicesConvergence = getTimeIndices(convergenceDFs, time)
    abundances_convergence = np.empty(shape=(nconvergence, nspecs))

    nparameters = len(parameterNames)
    statistics_converged = np.empty(shape=(nspecs, nparameters))

    convergenceParametersRIN = rankInverseNormalTransform(convergenceParameters)
    for i, spec in enumerate(specs):
        abundances_convergence[:, i] = getAbundances(
            convergenceDFs, timeIndicesConvergence, spec, on_grain=on_grain, njobs=njobs
        )
        abundances_convergence_RIN = rankInverseNormalTransform(
            abundances_convergence[:, i]
        )
        statistics_converged[i, :] = calculateAllCorrelations(
            abundances_convergence_RIN, convergenceParametersRIN, parameterNames
        )["statistic"].values

    nsamples = len(sampleDFs)
    timeIndicesSamples = getTimeIndices(sampleDFs, time)
    abundances_for_each_spec = np.empty(shape=(nsamples, nspecs))
    for i, spec in enumerate(specs):
        abundances_for_each_spec[:, i] = getAbundances(
            sampleDFs, timeIndicesSamples, spec, on_grain=on_grain, njobs=njobs
        )

    indeces = list(range(nsamples))
    num_samples = np.logspace(1, np.log10(nsamples), endpoint=True, dtype=int)

    statistics_for_nums = np.empty(shape=(nspecs, len(num_samples), nparameters))

    for i, num in enumerate(num_samples):
        shuffle(indeces)
        indeces_to_use = np.array(indeces)[:num]

        abundances_to_use = abundances_for_each_spec[indeces_to_use, :]
        abundances_to_use_RIN = rankInverseNormalTransform(abundances_to_use)

        parameters_to_use = sampleParameters[indeces_to_use, :]
        parameters_to_use_RIN = rankInverseNormalTransform(parameters_to_use)
        for j, spec in enumerate(specs):
            result_df = calculateAllCorrelations(
                abundances_to_use_RIN[:, j], parameters_to_use_RIN, parameterNames
            )

            statistics_for_nums[j, i, :] = result_df["statistic"].values

    # fig, axs = plt.subplots(1, nspecs, sharex=True, sharey=True)
    # for j, spec in enumerate(specs):
    #     axs[j].set_title(convertSpeciesToLegendLabel(spec))
    #     for k, parameterName in enumerate(parameterNames):
    #         axs[j].plot(num_samples, statistics_for_nums[j, :, k], c="k", alpha=0.1)
    # axs[j].set_xscale("log")
    # axs[0].set_ylabel("$r_{\mathrm{RIN}}\\left(N\\right)$")
    # fig.supxlabel("N")
    # plt.tight_layout()
    # plt.show()

    abs_diff = np.empty(shape=(nspecs, len(num_samples), nparameters))
    fig, axs = plt.subplots(1, nspecs, sharex=True, sharey=True)
    for j, spec in enumerate(specs):
        axs[j].set_title(convertSpeciesToLegendLabel(spec))
        abs_diff[j, :, :] = np.abs(
            statistics_for_nums[j, :, :] - statistics_converged[j, :]
        )
        for k, parameterName in enumerate(parameterNames):
            axs[j].plot(num_samples, abs_diff[j, :, k], c="k", alpha=0.1)

        confidence_interval = np.quantile(abs_diff[j, :, :], 0.95, axis=1)
        print(confidence_interval[-1])
        axs[j].plot(num_samples, confidence_interval)

    axs[j].set_xscale("log")
    axs[0].set_ylabel(
        f"$\\left|r_{{\mathrm{{RIN}}}}\\left(N\\right)-r_{{\mathrm{{RIN}}}}\\left({nconvergence}\\right)\\right|$"
    )
    axs[0].set_yscale("log")
    fig.supxlabel("$N$")
    plt.tight_layout()
    if savefig_path is not None:
        plt.savefig(savefig_path)
    plt.show()


if __name__ == "__main__":
    pass
