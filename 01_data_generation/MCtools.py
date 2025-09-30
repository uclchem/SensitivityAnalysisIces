import os
import sys
import time
from math import ceil
from subprocess import Popen, run

import numpy as np
import pandas as pd
import uclchem
from scipy.special import erf, erfinv
from scipy.stats.qmc import LatinHypercube, Sobol

from pathlib import Path

from scipy.constants import Boltzmann, atomic_mass, pi
from uclchem.makerates.species import Species
from uclchem.makerates.reaction import Reaction, CoupledReaction
from uclchem.makerates.network import Network

from typing import Literal

def generate_samples(
    network: list[Network],
    # species: list[Species],
    # reactions: list[Reaction],
    n_samples: int,
    # means: list[float],
    output_path: str | Path | None = None,
) -> np.ndarray:

    (
        species,
        reactions,
        binding_energies_surf,
        diffusion_barriers_surf,
        diffusionprefactors_surf,
        desorptionprefactors_surf,
        barriers_non_coupled,
    ) = extract_parameters_from_network(network)

    # Calculate desired standard deviations of normal distributions.
    standardDeviations = (
        [value_to_std(Ebind, type="energy") for Ebind in binding_energies_surf]
        + [value_to_std(Ediff, type="energy") for Ediff in diffusion_barriers_surf]
        + [
            value_to_std(prefac, type="prefactor")
            for prefac in diffusionprefactors_surf
        ]
        + [
            value_to_std(prefac, type="prefactor")
            for prefac in desorptionprefactors_surf
        ]
        + [value_to_std(Ereac, type="energy") for Ereac in barriers_non_coupled]
    )
    standardDeviations = np.array(standardDeviations)

    # Get desired means of normal distributions. Log-transform prefactors
    means = (
        binding_energies_surf
        + diffusion_barriers_surf
        + list(np.log10(diffusionprefactors_surf))
        + list(np.log10(desorptionprefactors_surf))
        + barriers_non_coupled
    )
    means = np.array(means)


    # Use regular Random sampling
    n_parameters = 4 * len(species) + len(reactions)
    uniformSamples = generateUniform(n_parameters, n_samples)

    # Convert the uniform distributions to truncated normal distributions
    normalSamples = inverseCDFBounded(
        uniformSamples, means, standardDeviations, 1, clip_minimum_to_zero=True
    )

    # Now we need to just transform the ones that were calculated in log10-space back to normal space
    firstPrefacIndex = len(binding_energies_surf) + len(diffusion_barriers_surf)
    lastPrefacIndex = n_parameters - len(barriers_non_coupled) - 1
    normalSamples[:, firstPrefacIndex : lastPrefacIndex + 1] = np.power(
        10, normalSamples[:, firstPrefacIndex : lastPrefacIndex + 1]
    )

    if output_path is not None:
        columns_binding_energies = [f"{str(spec)} bind" for spec in species]
        columns_diffusion_barriers = [f"{str(spec)} diff" for spec in species]
        columns_diff_prefacs = [f"{str(spec)} diffprefac" for spec in species]
        columns_des_prefacs = [f"{str(spec)} desprefac" for spec in species]
        columns_energy_barriers = [str(reaction) for reaction in reactions]
        columns_all = (
            columns_binding_energies
            + columns_diffusion_barriers
            + columns_diff_prefacs
            + columns_des_prefacs
            + columns_energy_barriers
        )
        # Write the parameters
        df = pd.DataFrame(normalSamples, columns=columns_all)
        df.to_csv(output_path, sep=",")
    return normalSamples


def value_to_std(value: float, type=Literal["energy", "prefactor"]) -> float:
    """Get the standard deviation to use for sampling different types of parameters

    Args:
        value (float): mean value of the parameter.
        type (str): whether the value is 'energy' or a 'prefactor'

    Returns:
        std (float): standard deviation to use for sampling
    """
    if type == "prefactor":
        return 2.0

    if value < 200:
        std = 100.0
    elif value < 1600:
        std = float(value) / 2.0
    else:
        std = 800.0
    return std


def extract_parameters_from_network(network: Network):
    # Get all LH reactions
    reacs = network.get_reaction_list()
    lh_reacs = [reac for reac in reacs if reac.get_reaction_type() == "LH"]

    # Remove all reactions that are so called "CoupledReaction" instances.
    # These are reactions like LHDES, or reactions happening in the bulk.
    # They do not need to be varied, as they are changed according to the corresponding
    # surface LH reactions.
    lh_reacs_non_coupled = [
        reaction for reaction in lh_reacs if not isinstance(reaction, CoupledReaction)
    ]

    # Get barriers of all the LH reactions
    barriers_non_coupled = [reaction.get_gamma() for reaction in lh_reacs_non_coupled]

    # Get list of all species
    species_list = network.get_species_list()

    # Get only the surface species
    species_surf = [
        spec
        for spec in species_list
        if spec.is_surface_species() and spec not in ["SURFACE", "BULK"]
    ]

    # Get binding energies of all surface species
    binding_energies_surf = [spec.binding_energy for spec in species_surf]

    # Calculate prefactors for all surface species
    diffusionprefactors_surf = [HHprefactor(spec) for spec in species_surf]
    desorptionprefactors_surf = [HHprefactor(spec) for spec in species_surf]

    # Calculate (using Ediff=0.5*Ebind) the nominal diffusion barriers of all surface species
    diffusion_barriers_surf = [
        spec.diffusion_barrier
        if spec.diffusion_barrier != 0.0
        else 0.5 * spec.binding_energy
        for spec in species_surf
    ]
    return (
        species_surf,
        lh_reacs_non_coupled,
        binding_energies_surf,
        diffusion_barriers_surf,
        diffusionprefactors_surf,
        desorptionprefactors_surf,
        barriers_non_coupled,
    )


def HHprefactor(species: Species) -> float:
    """Calculate the desorption prefactor, using the Hasegawa-Herbst (1992) formulation)

    Args:
        species (Species): Species instance

    Returns:
        prefactor (float): desorption prefactor in s-1
    """
    SURFACE_SITE_DENSITY = 1.5e15  # NUM SITES PER CM-2
    vdiff_prefactor = (
        2.0
        * float(Boltzmann)
        * 1e7
        * SURFACE_SITE_DENSITY
        / (pi**2 * atomic_mass * 1000)
    )
    prefactor = np.sqrt(vdiff_prefactor * species.binding_energy / species.mass)
    return prefactor


def recompile_UCLCHEM(
    uclchem_dir: str | Path | None = None, quiet: bool = True
) -> None:
    """Recompile UCLCHEM. Note that after recompilation, the UCLCHEM instance in this python instance is still the old one.
    To get the new instance, you need to execute 'python3 someFile.py' and you will have the new UCLCHEM there.

    Inputs:
        uclchem_dir (str | Path | None): directory of UCLCHEM. If None, tries to find it using get_UCLCHEM_dir
        quiet (bool): whether to suppress output during compilation and installation
    """
    if uclchem_dir is None:
        uclchem_dir = get_UCLCHEM_dir()
    cwd = os.getcwd()
    os.chdir(uclchem_dir)
    commands = f"{sys.executable} -m pip install --no-deps -q -e .".split()
    if not quiet:
        commands.pop(-3)
    run(commands, check=True)
    os.chdir(cwd)


def get_UCLCHEM_dir() -> Path:
    """Get the root directory of the UCLCHEM package

    Returns:
        uclchem_dir (str): directory of UCLCHEM
    """
    uclchem_dir = uclchem.__file__.split(os.path.sep)
    src_index = uclchem_dir.index("src")
    uclchem_dir = os.path.sep + os.path.join(*uclchem_dir[:src_index])
    return Path(uclchem_dir)


def isPowerOfTwo(x: int) -> bool:
    """Checks if x is a power of two

    Inputs:
        x (int): integer to check

    Returns:
        bool: whether x is a power of two
    """
    return (x != 0) and ((x & (x - 1)) == 0)


def generateSobol(
    dimensions: int, nsamples: int, usePowerOfTwo: bool = True
) -> np.ndarray:
    """Uniformly generate nsamples samples in dimensions dimensions using Sobol` sequences
    Generates samples in [0,1)^dimensions hypercube.

    Inputs:
        dimensions (int): number of dimensions
        nsamples (int): number of samples
        usePowerOfTwo (bool): whether to always sample a power of 2 samples.
            If true, increases the number of samples to the closest power of two,
            and then takes the first nsamples.

    Returns:
        samples (np.ndarray): array of shape nsamples*dimensions,
            continaining values between 0 and 1.
    """
    sobol = Sobol(dimensions)
    if usePowerOfTwo:
        if not isPowerOfTwo(nsamples):
            log2Samples = ceil(np.log2(nsamples))
        else:
            log2Samples = int(np.log2(nsamples))

        sobolSamples = sobol.random_base2(log2Samples)
        sobolSamples = sobolSamples[:nsamples]
    else:
        sobolSamples = sobol.random(nsamples)
    return sobolSamples


def generateLHS(dimensions: int, nsamples: int) -> np.ndarray:
    """Uniformly generate nsamples samples in dimensions dimensions using Latin Hypercube Sampling
    Generates samples in [0,1)^dimensions hypercube.

    Inputs:
        dimensions (int): number of dimensions
        nsamples (int): number of samples

    Returns:
        samples (np.ndarray): array of shape nsamples*dimensions,
            continaining values between 0 and 1.
    """
    lhs = LatinHypercube(dimensions)
    samples = lhs.random(nsamples)
    return samples


def generateUniform(dimensions: int, nsamples: int) -> np.ndarray:
    """Uniformly generate nsamples samples in dimensions dimensions.
    Generates samples in [0,1)^dimensions hypercube.

    Inputs:
        dimensions (int): number of dimensions
        nsamples (int): number of samples

    Returns:
        samples (np.ndarray): array of shape nsamples*dimensions,
            continaining values between 0 and 1.
    """
    samples = np.random.uniform(size=(nsamples, dimensions))
    return samples


def inverseCDF(
    samples: float | np.ndarray,
    mus: float | np.ndarray,
    sigmas: float | np.ndarray,
) -> float | np.ndarray:
    """Inverse cumulative distribution function of normal distribution

    Inputs:
        samples (float | np.ndarray): uniformly distributed samples
        mus (float | np.ndarray): desired mean
        sigmas (float | np.ndarray): desired standard deviation

    Returns:
        transformedSamples (float | np.ndarray): transformed samples
    """
    transformedSamples = mus + sigmas * np.sqrt(2) * erfinv(2 * samples - 1)
    return transformedSamples


def CDFnormal(x: float | np.ndarray, mu: float, sigma: float) -> float | np.ndarray:
    """Cumulative distribution function of normal distribution

    Inputs:
        x (float | np.ndarray): positions where to evaluate CDF
        mu (float): mean of normal distribution
        sigma: (float): standard deviation of normal distribution

    Returns:
        float | np.ndarray: CDF at positions x
    """
    return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def inverseCDFBounded(
    samples: float | np.ndarray,
    mus: float | np.ndarray,
    sigmas: float | np.ndarray,
    bounds: float,
    clip_minimum_to_zero: bool = True,
) -> float | np.ndarray:
    """Inverse cumulative distribution function of a bounded normal distribution.
    Clips values of mus - sigmas to at least 0.

    Inputs:
        samples (float | np.ndarray): uniformly distributed samples
        mus (float | np.ndarray): desired mean
        sigmas (float | np.ndarray): desired standard deviation
        bounds (float): bounds (in units of sigma). If 0, unbounded.
        clip_minimum_to_zero (bool) = whether to clip the minimum to 0. Does nothing if unbounded.

    Returns:
        float | np.ndarray: transformed samples
    """
    # Clip values to 0 at least.
    if bounds > 0:
        xlower = mus - bounds * sigmas
        if clip_minimum_to_zero:
            xlower = np.clip(xlower, 0, None)

        xupper = mus + bounds * sigmas

        CDFatLower = CDFnormal(xlower, mus, sigmas)
        CDFatUpper = CDFnormal(xupper, mus, sigmas)

        downshift = CDFatLower
        scaling = CDFatUpper - CDFatLower
        p = scaling * samples + downshift

        return inverseCDF(p, mus, sigmas)
    if bounds == 0:
        return inverseCDF(samples, mus, sigmas)
    else:
        raise ValueError(
            f"bounds needs to be a positive integer or 0, but was {bounds}"
        )


def readMCparameters(filepath: str) -> tuple[np.ndarray]:
    """Read the MC parameters generated by previous runs

    Inputs:
        filepath (str): filepath to csv file

    Returns:
        bindingEnergies (np.ndarray): array containing all binding energy samples
        diffusionBarriers (np.ndarray): array containing all diffusion barrier samples
        energyBarriers (np.ndarray): array containing all reaction energy barrier samples
        diffusionPrefactors (np.ndarray): array containing all diffusion prefactor samples
        desorptionPrefactors (np.ndarray): array containing all desorption prefactor samples
    """
    df = pd.read_csv(filepath)
    parameterNames = df.columns
    bindingEnergies = np.array(
        [df[param].to_numpy() for param in parameterNames if "bind" in param]
    )
    diffusionBarriers = np.array(
        [
            df[param].to_numpy()
            for param in parameterNames
            if ("diff" in param and "prefac" not in param)
        ]
    )
    energyBarriers = np.array(
        [df[param].to_numpy() for param in parameterNames if "->" in param]
    )
    diffusionPrefactors = np.array(
        [df[param].to_numpy() for param in parameterNames if "diffprefac" in param]
    )
    desorptionPrefactors = np.array(
        [df[param].to_numpy() for param in parameterNames if "desprefac" in param]
    )
    return (
        bindingEnergies,
        diffusionBarriers,
        energyBarriers,
        diffusionPrefactors,
        desorptionPrefactors,
    )


def remove_duplicates(lst: list) -> list:
    """Remove duplicates of a list, while keeping the order

    Inputs:
        lst (list): list of objects

    Returns:
        list: input list but without duplicates
    """
    seen = set()
    seen_add = seen.add
    return [x for x in lst if not (x in seen or seen_add(x))]


def create_grid(
    column_names: list[str], *params: list[list[float]], grid_folder: str = "."
) -> pd.DataFrame:
    """Create a grid of physical conditions to run. Makes a dense grid.

    Inputs:
        column_names (list[str]): names of the columns for physical parameters
        *params (list[list[float]]): list of physical conditions.
        grid_folder (str): directory where to put the model outputs

    Returns:
        model_table (pd.DataFrame): dataframe containing all the models
    """
    n_params = len(column_names)
    assert len(params) == n_params, (
        "Number of column names and input parameters not the same"
    )
    params = [remove_duplicates(param_lst) for param_lst in params]
    parameterSpace = np.asarray(np.meshgrid(*params)).reshape(n_params, -1)
    model_table = pd.DataFrame(parameterSpace.T, columns=column_names)
    model_table["outputFile"] = model_table.apply(
        lambda row: f"{grid_folder}/{'_'.join([str(row[column_names[i]]) for i in range(n_params)])}",
        axis=1,
    )
    return model_table


def combine_grids(grids: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine multiple grids into one grid

    Inputs:
        grids (list[pd.DataFrame]): list of grids.

    Returns:
        combined_grid (list[pd.DataFrame]): combined grid
    """
    combined_grid = pd.concat(grids, ignore_index=True)
    return combined_grid


def get_running_processes(processes: list[Popen]) -> list[Popen]:
    """Get the processes that are still running from a list of processes

    Inputs:
        processes (list[Popen]): list of processes

    Returns:
        running_processes (list[Popen]): list of running processes
    """
    running_processes = [process for process in processes if process.poll() is None]
    return running_processes


def terminate_all_processes(processes: list[Popen]) -> None:
    """Terminate any process that is still running from a list of processes.

    Inputs:
        processes (list[Popen]): list of processes
    """
    processes = get_running_processes(processes)
    for proc in processes:
        proc.terminate()


def run_grid(
    grid_table: pd.DataFrame,
    processes: list[Popen],
    run_model_file: str,
    sample_nr: int | str,
    n_jobs: int,
) -> list[Popen]:
    """Run a grid of physical conditions, while keeping track of the number of running jobs.

    Inputs:
        grid_table (pd.DataFrame): dataframe with physical conditions to run
        processes (list[Popen]): list of processes
        run_model_file (str): python file to run that executes a model.
        sample_nr (int | str): sample number
        n_jobs (int): number of jobs to run at once

    Returns:
        processes (list[Popen]): list of processes (including given initially)
    """
    for j, row in grid_table.iterrows():
        # Manage the amount of running processes
        while len(get_running_processes(processes)) >= n_jobs:
            time.sleep(1)

        print(
            f"Starting model run {sample_nr} with physical conditions {row['temperature']} {row['density']} {row['zeta']} {row['radfield']} (physical conditions row index {j})"
        )
        model_run_process = Popen(
            [
                f"{sys.executable} {run_model_file} {row['outputFile']} {sample_nr} {row['temperature']} {row['density']} {row['zeta']} {row['radfield']}"
            ],
            shell=True,
            stdin=None,
            stdout=None,
            stderr=None,
            close_fds=True,
        )
        # Append model run to list of processes
        processes.append(model_run_process)
    return processes
