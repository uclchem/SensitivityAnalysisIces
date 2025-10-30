from __future__ import annotations

import itertools
import math
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from glob import glob
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from pathlib import Path
from random import shuffle

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

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


def set_environ_njobs(njobs: int | str) -> None:
    os.environ["OMP_NUM_THREADS"] = str(njobs)
    os.environ["OPENBLAS_NUM_THREADS"] = str(njobs)
    os.environ["MKL_NUM_THREADS"] = str(njobs)
    os.environ["VECLIB_NUM_THREADS"] = str(njobs)
    os.environ["NUMEXPR_NUM_THREADS"] = str(njobs)


@dataclass
class Style:
    zero_correlation_ls = "dashed"
    zero_correlation_lw = 1.0
    zero_correlation_color = "gray"
    zero_correlation_alpha = 0.6

    weak_correlation_color = "gray"
    weak_correlation_alpha = 0.15

    nominal_color = "#FFB8F2"
    nominal_ls = "dashed"
    average_color = "#8EFF8B"
    average_ls = "solid"
    sample_color = "black"
    sample_lw = 0.2
    sample_alpha = 0.1

    min_statistic = 0.4
    correlation_ci_alpha = 0.25
    correlation_ci_edgecolor = "face"
    correlation_ci_lw = 0.15
    sampling_95_ci = 0.08

    ylim = [1e-16, 1e-2]
    xlim = [1e0, 1e6]

    marker_edgelinewidth = 0.1
    marker_edgecolor = "black"
    abundance_marker_size = 12
    abundance_marker_alpha = 0.75
    nominal_marker_size = 50

    abundance_label = "Abundance (wrt H)"
    time_label = "Time (years)"
    temperature_label = "Temperature (K)"
    rRIN_label = "$r_{\mathrm{RIN}}$"

    title_pad = 3.5


def set_rc_params(font: str = "AA") -> None:
    font = font.lower()
    if font not in ["aa", "cmu_bright"]:
        raise ValueError()

    plt.style.use(Path("style.mplstyle"))

    if font == "cmu_bright":
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = "CMU Bright"
        mpl.rcParams["axes.unicode_minus"] = False
        mpl.rcParams["text.usetex"] = False

        # The "\usepackage[OT1]{fontenc}" ensures that we do not use oldstyle numbers, which
        # does not look great imo.
        mpl.rcParams["text.latex.preamble"] = (
            r"\usepackage{cmbright}\n\usepackage[OT1]{fontenc}"
        )
        mpl.rcParams["mathtext.fontset"] = "custom"
        mpl.rcParams["mathtext.cal"] = "sans:italic"
        mpl.rcParams["mathtext.default"] = "it"
        mpl.rcParams["pdf.fonttype"] = 42


class DataManager:
    def __init__(
        self, directory: str | Path, style: Style | None = None, njobs: int = 1
    ):
        """Convenient way to get all the files at different physical conditions

        Args:
            directory (str | Path): directory containing all the ".h5" files
            style (Style): style dataclass containing all plotting parameters for consistency
            njobs (int): number of jobs to use to read files and get abundances.
        """
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"{self.directory} is not a valid directory")
        self.parameters_path = self.directory / "MC_parameter_runs.csv"

        parametersDF = pd.read_csv(self.parameters_path, index_col=0)
        self.parameters = parametersDF.to_numpy()

        self.parameter_names = parametersDF.columns.to_numpy()
        self.parameters_RIN = rankInverseNormalTransform(self.parameters)

        # Get filepaths of all model runs
        self.filepaths_samples, self.filepaths_nominal = getAllRunsFilepaths(
            self.directory, extension="h5"
        )
        self.from_single_hdf_files = self.filepaths_nominal is None
        self.physical_conditions = getPhysicalParamSets(
            [filepaths[0] for filepaths in self.filepaths_samples]
        )

        if style is None:
            style = Style()
        self.style = style
        self.njobs = njobs

    def get_total_runtime(self) -> float:
        """Get total runtime.

        Returns:
            float: total run of all calculations in this directory
        """
        return getTotalRuntime(list(itertools.chain(*self.filepaths_samples)))

    def get_physical_conditions_index(
        self, T: float, nH: float, zeta: float, radfield: float
    ) -> int:
        """Get physical condition index of a certain set of physical conditions in this directory.

        Args:
            T (float): temperature in K
            nH (float): number density in cm-3
            zeta (float): cosmic ray ionization rate as multiple of 1.3*10^{-17} s-1
            radfield (float): UV field strength in Habing

        Returns:
            int: index of set of physical conditions in list of filenames.
        """
        if T is None or nH is None or zeta is None or radfield is None:
            raise ValueError()
        try:
            return physicalParamSetToIndex(
                self.physical_conditions, T=T, nH=nH, zeta=zeta, radfield=radfield
            )
        except ValueError as e:
            raise ValueError(
                f"T={T} K, nH={nH} cm-3, zeta={zeta}, radfield={radfield} not in list of physical conditions.\nList of physical conditions:\n{self.physical_conditions}"
            ) from e

    def get_dataframes(
        self,
        T: float = None,
        nH: float = None,
        zeta: float = None,
        radfield: float = None,
        load_rates: bool = False,
    ) -> Models:
        """Get Models instance corresponding to specific set of physical conditions

        Args:
            T (float): temperature in K
            nH (float): number density in cm-3
            zeta (float): cosmic ray ionization rate as multiple of 1.3*10^{-17} s-1
            radfield (float): UV field strength in Habing
            load_rates (bool): Whether to also load the reaction rates. Default: False

        Returns:
            Models: Models instance containing all dataframes.
        """

        index = self.get_physical_conditions_index(T, nH, zeta, radfield)

        # Get filepaths of all model runs at this certain set of physical conditions
        if self.from_single_hdf_files:
            return Models.from_single_hdf(
                self.filepaths_samples[index][0], load_rates=load_rates
            )
        sampleDFs, nominalDF = getDataFramesForPhysicalParamSet(
            index, self.filepaths_samples, self.filepaths_nominal, njobs=self.njobs
        )
        return Models(sampleDFs, nominalDF)

    def get_parameter_values(self, parameter_name: str, get_RIN: bool = False):
        """Get the values of a certain parameter from the MC_parameter_runs file.

        Args:
            parameter_name (str): column name of the parameter, e.g. "#H diff"
            get_RIN (bool): whether to get the rank-inverse normal transformed values. Default: False

        Returns:
            np.ndarray: values of that parameter
        """
        index = np.nonzero(self.parameter_names == parameter_name)[0]
        if get_RIN:
            return self.parameters_RIN[:, index]
        else:
            return self.parameters[:, index]

    def create_standard_plot(
        self,
        T: float,
        nH: float,
        zeta: float,
        radfield: float,
        species: list[str],
        times=np.logspace(0, 6, num=100),
        on_grain: bool = False,
        save_fig_path: str | Path | None = None,
        plot_individual_samples: bool = True,
        colors: list[str] | None = None,
        put_legend_on_side: bool = False,
    ) -> None:
        models = self.get_dataframes(
            T=T, nH=nH, zeta=zeta, radfield=radfield, load_rates=False
        )

        models = models.get_models_that_reach_desired_time(times[-1])

        if models.nsamples < np.shape(self.parameters_RIN)[0]:
            print(
                "WARNING: Number of samples run is smaller than number of rows in parameters file."
            )
            print(
                f"\tAssuming that the model samples are run in order. Using the first {models.nsamples} rows"
            )
            # The number of models that have been run is smaller than the number
            # models in the final parameters.
            # Assume they are ordered.
            parameters_RIN = rankInverseNormalTransform(
                self.parameters[: models.nsamples, :]
            )
        else:
            parameters_RIN = self.parameters_RIN

        models.set_times(times)
        models.create_standard_plot(
            species,
            parameters_RIN,
            self.parameter_names,
            on_grain=on_grain,
            njobs=self.njobs,
            save_fig_path=save_fig_path,
            style=self.style,
            plot_individual_samples=plot_individual_samples,
            colors=colors,
            put_legend_on_side=put_legend_on_side,
        )

    def create_big_plot(
        self,
        temps: list[float],
        densities: list[float],
        zeta: float,
        radfield: float,
        species: list[str],
        times=np.logspace(0, 6, num=100),
        on_grain: bool = False,
        save_fig_path: str | Path | None = None,
        plot_individual_samples: bool = True,
        colors: list[str] | None = None,
    ) -> None:
        if colors is None:
            colors = getColors()
        # suptitle_h_loc = 0.55

        fig = plt.figure(figsize=(8, 8))
        outer_gs = fig.add_gridspec(len(temps), len(densities))
        seenSpeciesParameters = []

        axsStorage = np.empty(
            shape=(len(temps), len(densities), 2, len(species)), dtype=object
        )
        suplabels = []
        for i, temp in enumerate(temps):
            for j, density in enumerate(densities):
                print(temp, density)
                seenParameters = []
                log10Density = int(np.log10(density))
                title = f"$T={int(temp)}$ K, $n_{{\mathrm{{H}}}}=10^{{{log10Density}}}$ cm$^{{-3}}$"
                subfig = fig.add_subfigure(outer_gs[i, j])
                subfig.suptitle(title, y=0.98, fontsize=12)
                gs = subfig.add_gridspec(
                    nrows=2,
                    ncols=len(species),
                    wspace=0,
                    hspace=0,
                    height_ratios=(1.0, 2.0 / 3.0),
                )
                gs.update(left=0.05, bottom=0.13, top=0.91)

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
                    axs[0, k].set_xlim(self.style.xlim)
                    axs[0, k].set_ylim(self.style.ylim)
                    axs[0, k].set_yscale("log")
                    axs[0, k].set_title(
                        convertSpeciesToLegendLabel(spec),
                        y=0.80,
                    )

                    axs[1, k].set_ylim([-1.2, 1.2])
                    axs[1, k].axhline(
                        0,
                        c=self.style.zero_correlation_color,
                        alpha=self.style.zero_correlation_alpha,
                        ls=self.style.zero_correlation_ls,
                        lw=self.style.zero_correlation_lw,
                    )
                    axs[1, k].fill_between(
                        [0, 1e10],
                        [-self.style.min_statistic] * 2,
                        [self.style.min_statistic] * 2,
                        color=self.style.weak_correlation_color,
                        alpha=self.style.weak_correlation_alpha,
                        edgecolor="none",
                    )
                axs[0, 0].set_ylabel(self.style.abundance_label)
                axs[1, 0].set_ylabel(self.style.rRIN_label)

                suplabel = subfig.supxlabel(
                    self.style.time_label
                )  # , x=suptitle_h_loc, y=-0.015)
                suplabels.append(suplabel)

                axs[0, 0].set_xticks([1e0, 1e2, 1e4, 1e6])
                [axs[0, l].set_xticks([1e2, 1e4, 1e6]) for l in range(1, len(species))]
                [
                    axs[0, l].set_xticks([1e1, 1e3, 1e5], ["", "", ""], minor=True)
                    for l in range(len(species))
                ]
                axsStorage[i, j, :, :] = axs

                models = self.get_dataframes(
                    T=temp, nH=density, zeta=zeta, radfield=radfield, load_rates=False
                )
                models = models.get_models_that_reach_desired_time(times[-1])

                if np.shape(self.parameters_RIN)[0] == models.nsamples:
                    parameters_RIN = self.parameters_RIN
                else:
                    print("WARNING")
                    parameters_RIN = rankInverseNormalTransform(
                        self.parameters[: models.nsamples, :]
                    )

                models.set_times(times)

                for k, spec in enumerate(species):
                    # Lightly plot all model runs
                    models.plot_abundances_panel(
                        spec,
                        axs[0, k],
                        on_grain=on_grain,
                        plot_individual_samples=plot_individual_samples,
                        njobs=self.njobs,
                        style=self.style,
                    )

                    abundances_RIN = models.get_abundances_RIN(
                        spec, on_grain=on_grain, njobs=self.njobs
                    )

                    # Perform rank-based inverse normal transformation on the abundances
                    sigCorrelations = calculateSignificantCorrelations2D(
                        abundances_RIN,
                        parameters_RIN,
                        self.parameter_names,
                        confidence_level=0.95,
                        minStatistic=self.style.min_statistic,
                        calculate_confidence_interval=True,
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

                        paramType = get_param_type_from_param(row["parameter"])
                        linestyle = get_linestyle_from_param_type(paramType)
                        speciesParameter = get_species_from_param(row["parameter"])

                        if speciesParameter not in seenSpeciesParameters:
                            seenSpeciesParameters.append(speciesParameter)
                        colorIndex = (
                            seenSpeciesParameters.index(speciesParameter)
                        ) % len(colors)

                        axs[1, k].plot(
                            models.times,
                            row["statistic"],
                            c=colors[colorIndex],
                            ls=linestyle,
                            **plot_kwargs,
                        )

                        axs[1, k].fill_between(
                            models.times,
                            np.array(row["cilow"]) - self.style.sampling_95_ci,
                            np.array(row["cihigh"]) + self.style.sampling_95_ci,
                            alpha=self.style.correlation_ci_alpha,
                            color=colors[colorIndex],
                            edgecolor=self.style.correlation_ci_edgecolor,
                            linewidth=self.style.correlation_ci_lw,
                        )

        legendHandles = [
            Line2D([0], [0], c=colors[(i) % len(colors)])
            for i in range(len(seenSpeciesParameters))
        ]
        leg = axsStorage[0, 0, 0, 0].legend(
            legendHandles,
            [i for i in seenSpeciesParameters],
            loc="center left",
            bbox_to_anchor=(0.0, 1.27),
            handlelength=0,
            labelcolor="linecolor",
            handletextpad=0,
            labelspacing=0.4,
            columnspacing=1.5,
            ncols=6,
        )

        [handle.set_visible(False) for handle in leg.legend_handles]
        axsStorage[0, 0, 0, 0].add_artist(leg)
        leg.set_in_layout(False)

        paramTypesLegend = [
            r"$E_{\mathrm{diff}}$",
            r"$\nu_{\mathrm{diff}}$",
            r"$E_{\mathrm{bind}}$",
            r"$\nu_{\mathrm{des}}$",
            r"$E_{\mathrm{reac}}$",
        ]
        ls = ["solid", "dashdot", "dashed", "dotted", (0, (5, 1))]
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

        if save_fig_path is None:
            plt.show()
        else:
            plt.savefig(save_fig_path, bbox_extra_artists=[leg, leg2, *suplabels])
            plt.close()

    def create_standard_plots_for_all_physical_conditions(
        self,
        species,
        times=np.logspace(0, 6, 100),
        on_grain: bool = False,
        prefix: str = "sensitivities_",
        postfix=".pdf",
        filter_zeta: list[float] | float | None = None,
        filter_radfield: list[float] | float | None = None,
        plot_individual_samples: bool = True,
        put_legend_on_side: bool = False,
    ) -> None:
        mpl.use("agg")
        for physical_condition in self.physical_conditions:
            if filter_zeta is not None:
                if isinstance(filter_zeta, float):
                    if physical_condition[2] != filter_zeta:
                        continue
                else:
                    if physical_condition[2] not in filter_zeta:
                        continue
            if filter_radfield is not None:
                if isinstance(filter_radfield, float):
                    if physical_condition[3] != filter_radfield:
                        continue
                else:
                    if physical_condition[3] not in filter_radfield:
                        continue

            physical_condition_string = physicalParamSetToSaveString(physical_condition)
            path = f"{prefix}{physical_condition_string}{postfix}"
            self.create_standard_plot(
                *physical_condition,
                species,
                times=times,
                on_grain=on_grain,
                save_fig_path=path,
                plot_individual_samples=plot_individual_samples,
                put_legend_on_side=put_legend_on_side,
            )
            plt.close()

    def plot_abundances_against_parameter(
        self,
        T: float,
        nH: float,
        zeta: float,
        radfield: float,
        species: list[str],
        parameter_name: str,
        times: list[float] | float,
        on_grain: bool = False,
        save_fig_path: str | Path | None = None,
        xticks: tuple[float] | None = None,
    ) -> None:
        models = self.get_dataframes(
            T=T,
            nH=nH,
            zeta=zeta,
            radfield=radfield,
            load_rates=False,
        )
        if isinstance(times, float):
            times = [times]
        models = models.get_models_that_reach_desired_time(times[-1])
        models.set_times(times)
        parameter_values = self.get_parameter_values(parameter_name, get_RIN=False)
        if np.shape(parameter_values) == models.nsamples:
            parameter_RIN = self.get_parameter_values(parameter_name, get_RIN=True)
        else:
            parameter_values = parameter_values[: models.nsamples]
            parameter_RIN = rankInverseNormalTransform(parameter_values)
        models.plot_abundances_against_parameter(
            species,
            parameter_values,
            parameter_RIN,
            parameter_name,
            on_grain=on_grain,
            njobs=self.njobs,
            save_fig_path=save_fig_path,
            style=self.style,
            xticks=xticks,
        )

    def create_widths_plot(
        self,
        temps: list[float],
        densities: list[float],
        zeta: float,
        radfield: float,
        species: list[str],
        confidence_level: float = 0.6827,
        times: list[float] = np.logspace(0, 6, num=100),
        on_grain: bool = False,
        save_fig_path: str | Path | None = None,
    ) -> None:
        fig, axs = plt.subplots(
            1,
            len(species),
            figsize=(5, 2.598425197),
            sharex=False,
            sharey=True,
        )
        widths_at_densities = np.zeros((len(species), len(densities), len(times)))

        temp_diff = temps[1] - temps[0]
        color_bounds = [temp - 0.5 * temp_diff for temp in temps] + [
            temps[-1] + 0.5 * temp_diff
        ]

        temp_colors = sns.color_palette("flare_r", as_cmap=True)
        # temp_colors = sns.color_palette("rocket", as_cmap=True)
        # temp_colors.colors = temp_colors.colors[50:175]
        norm = mpl.colors.BoundaryNorm(color_bounds, temp_colors.N)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=temp_colors)

        for k, temp in enumerate(temps):
            for j, density in enumerate(densities):
                models = self.get_dataframes(
                    T=temp, nH=density, zeta=zeta, radfield=radfield, load_rates=False
                )
                models.get_models_that_reach_desired_time(times[-1])
                models.set_times(times)
                times = models.times

                for i, spec in enumerate(species):
                    cilow, cihigh = models.get_confidence_interval_of_abundances(
                        spec,
                        confidence_level=confidence_level,
                        on_grain=on_grain,
                        njobs=self.njobs,
                    )
                    width = cihigh / cilow
                    widths_at_densities[i, j, : len(width)] = width

            for i, spec in enumerate(species):
                averageWidthOverDensities = np.power(
                    10,
                    np.average(
                        np.log10(widths_at_densities[i, :, : len(width)]), axis=0
                    ),
                )
                axs[i].plot(
                    times,
                    averageWidthOverDensities,
                    c=temp_colors(k * 1.0 / (len(temps) - 1)),
                    zorder=2,
                )

        for i, spec in enumerate(species):
            axs[i].set_title(
                convertSpeciesToLegendLabel(spec), pad=self.style.title_pad
            )

            # Draw line at 2 orders of magnitude.
            axs[i].axhline(
                1e2,
                zorder=1,
                c=self.style.zero_correlation_color,
                ls=self.style.zero_correlation_ls,
                alpha=self.style.zero_correlation_alpha,
                lw=self.style.zero_correlation_lw,
            )

        fig.supxlabel(self.style.time_label)
        for ax in axs.flat:
            ax.set_xlim([1e0, 1e6])
            ax.set_xscale("log")
            ax.label_outer()
        # Set xticks
        axs[0].set_xticks([1e0, 1e2, 1e4, 1e6])
        [axs[i].set_xticks([1e2, 1e4, 1e6]) for i in range(1, len(species))]
        [
            axs[i].set_xticks([1e1, 1e3, 1e5], ["", "", ""], minor=True)
            for i in range(len(species))
        ]

        # Set bottom of plot to 1
        axs[0].set_ylim(bottom=1e0)
        axs[0].set_yscale("log")
        axs[0].set_ylabel(f"Width of {confidence_level * 100}\% confidence interval")

        plt.subplots_adjust(wspace=0, bottom=0.13, right=0.81)
        cbar_ax = fig.add_axes([0.83, 0.13, 0.025, 0.75])
        cb = fig.colorbar(sm, ticks=temps, cax=cbar_ax)
        cb.ax.tick_params(axis="y", direction="out")
        cb.ax.minorticks_off()
        cb.set_label(self.style.temperature_label)

        if save_fig_path is not None:
            plt.savefig(save_fig_path)
            plt.close()
        else:
            plt.show()


class Models:
    def __init__(
        self,
        sample_dfs: list[pd.DataFrame],
        nominal_df: pd.DataFrame,
        times: np.ndarray | None = None,
        rate_df: pd.DataFrame | None = None,
    ):
        """Convenient way to keep track of all the samples and nominal run at one set of physical conditions

        Args:
            sample_dfs (list[pd.DataFrame]): list of sample dataframes
            nominal_df (pd.DataFrame): dataframe of model run with nominal network
            times (np.ndarray | None): times at which we want to determine abundances. Default: None
            rate_df (pd.DataFrame | None): dataframe containing reaction rates at every timestep. Default: None
        """
        self.sample_dfs = sample_dfs
        self.nominal_df = nominal_df
        self.rate_df = rate_df
        self.nsamples = len(self.sample_dfs)

        self.abundances = {}

        if times is None:
            self.times = None
            return

        self.set_times(times)

    @classmethod
    def from_single_hdf(
        cls,
        filepath: str | Path,
        times: np.ndarray | None = None,
        load_rates: bool = True,
    ):
        """Get the Models instance directly from a single hdf5 file containing all necessary information.

        Args:
            filepath (str | Path): filepath of hdf5 file
            times (np.ndarray | None): times at which we want to determine abundances. Default: None
            load_rates (bool): whether to load the reaction rates too. Default: True

        Returns:
            Models: instance of Models
        """
        with h5py.File(filepath, "r") as file:
            abund_cols = [col.decode("utf-8") for col in file["abundances_columns"]]
            nominal_df = pd.DataFrame(file["nominal"], columns=abund_cols)

            if load_rates and "rates_columns" in file.keys():
                rate_cols = [col.decode("utf-8") for col in file["rates_columns"]]
                nominal_rates_df = pd.DataFrame(
                    file["nominal_rates"], columns=rate_cols
                )
            else:
                nominal_rates_df = None

            sample_dfs = []
            sample_keys = [key for key in file.keys() if key.isdigit()]
            for i in range(1000):
                if str(i) not in sample_keys:
                    raise ValueError(i)
            sample_keys.sort(key=lambda x: int(x))
            for sample_key in sample_keys:
                sample_dfs.append(pd.DataFrame(file[sample_key], columns=abund_cols))

            return cls(sample_dfs, nominal_df, times=times, rate_df=nominal_rates_df)

    def set_times(self, times: list[float] | float) -> None:
        """Set the times at which we want to determine the abundances

        Args:
            times (list[float] | float): list of times or single time at which we want to determine the abundances.
        """
        if isinstance(times, int):
            times = float(times)
        if isinstance(times, float):
            times = [times]

        self.sample_time_indeces = getTimeIndices(self.sample_dfs, times)
        self.nominal_time_indeces = getTimeIndices([self.nominal_df], times)

        # Also get the times that those indices correspond to, such that we can plot them later.
        timeArray = getTimeFromIndices(self.sample_dfs, self.sample_time_indeces)

        self.sample_time_indeces, timeArray = checkTimeIndices(
            self.sample_dfs, timeArray, self.sample_time_indeces, times
        )
        self.times = timeArray[0, :]

    def get_abundances(
        self,
        spec: str,
        on_grain: bool = False,
        njobs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the abundances at the specified times for a certain species.

        Args:
            spec (str): Species, indicated by string. For example, "H" would be atomic hydrogen.
            on_grain (bool): Whether to return abundance on grain (i.e. surface + bulk abundance)
            njobs (int): number of jobs to use to get abundances

        Returns:
            sample_abundances (np.ndarray): Ntimes * Nsamples array of abundances in all samples
            nominal_abundances (np.ndarray): Ntimes array of abundances in nominal run
        """
        spec_key = self.spec_name_to_abundances_key(spec, on_grain)

        if spec_key not in self.abundances:
            sample_abundances = getAbundances(
                self.sample_dfs,
                self.sample_time_indeces,
                spec=spec,
                on_grain=on_grain,
                njobs=njobs,
            )
            nominal_abundances = getAbundance(
                self.nominal_df,
                range(len(self.nominal_df.index)),
                spec=spec,
                on_grain=on_grain,
            )
            self.abundances[spec_key] = (sample_abundances, nominal_abundances)

        return self.abundances[spec_key]

    def get_abundances_RIN(
        self,
        spec: str,
        on_grain: bool = False,
        njobs: int = 1,
    ) -> np.ndarray:
        """Get the rank-inverse normal transformed abundances in the samples at the specified times for a certain species.

        Args:
            spec (str): Species, indicated by string. For example, "H" would be atomic hydrogen.
            on_grain (bool): Whether to return abundance on grain (i.e. surface + bulk abundance)
            njobs (int): number of jobs to use to get abundances

        Returns:
            np.ndarray: Ntimes * Nsamples array of RIN-transformed abundances in all samples
        """
        samples, _ = self.get_abundances(spec, njobs=njobs, on_grain=on_grain)
        return rankInverseNormalTransform(samples)

    def spec_name_to_abundances_key(self, spec: str, on_grain: bool) -> str:
        """Convert the species name and whether it is on_grain to a key in self.abundances dictionary.
        This is used for caching abundances, so we don't need to determine them multiple times.

        Args:
            spec (str): Species, indicated by string. For example, "H" would be atomic hydrogen.
            on_grain (bool): Whether to get abundance key for on grain species.

        Returns:
            str: key in self.abundances dictionary. String representation of species + "_ice" or "_gas"
        """
        if on_grain:
            return spec + "_ice"
        else:
            return spec + "_gas"

    def get_models_that_reach_desired_time(self, desired_time: float) -> Models:
        return Models(
            [
                sample_df
                for sample_df in self.sample_dfs
                if sample_df["Time"].iloc[-1] >= desired_time
            ],
            self.nominal_df,
            times=self.times,
        )

    def get_log_average_abundance(
        self, spec: str, on_grain: bool = False, njobs: int = 1
    ) -> np.ndarray:
        """Get log-average of abundances of a certain species at all timepoints

        Args:
            spec (str): Species, indicated by string. For example, "H" would be atomic hydrogen.
            on_grain (bool): Whether to return log-average of abundance on grain (i.e. surface + bulk abundance)
            njobs (int): number of jobs to use to get abundances

        Returns:
            np.ndarray: Ntimes array of log-average of all samples
        """
        samples, _ = self.get_abundances(spec, on_grain=on_grain, njobs=njobs)
        return np.power(10, np.average(np.log10(samples), axis=0))

    def get_median_abundance(
        self, spec: str, on_grain: bool = False, njobs: int = 1
    ) -> np.ndarray:
        """Get median of abundances of a certain species at all timepoints

        Args:
            spec (str): Species, indicated by string. For example, "H" would be atomic hydrogen.
            on_grain (bool): Whether to return median of abundance on grain (i.e. surface + bulk abundance)
            njobs (int): number of jobs to use to get abundances

        Returns:
            np.ndarray: Ntimes array of median of all samples
        """
        samples, _ = self.get_abundances(spec, on_grain=on_grain, njobs=njobs)
        return np.median(samples, axis=0)

    def get_confidence_interval_of_abundances(
        self,
        spec: str,
        confidence_level: float = 0.6827,
        on_grain: bool = False,
        njobs: int = 1,
    ) -> tuple[np.ndarray]:
        """Get confidence interval of abundances across samples of a certain species at all timepoints

        Args:
            spec (str): Species, indicated by string. For example, "H" would be atomic hydrogen.
            confidence_level (float): confidence level (between 0 and 1). Default: 0.6827
            on_grain (bool): Whether to return confidence interval of abundance on grain (i.e. surface + bulk abundance)
            njobs (int): number of jobs to use to get abundances

        Returns:
            cilow (np.ndarray): lower limit of confidence interval
            cihigh (np.ndarray): upper limit of confidence interval
        """
        samples, _ = self.get_abundances(spec, on_grain=on_grain, njobs=njobs)
        cilow = np.quantile(samples, (1.0 - confidence_level) / 2.0, axis=0)
        cihigh = np.quantile(samples, (1.0 + confidence_level) / 2.0, axis=0)
        return cilow, cihigh

    def create_standard_plot(
        self,
        species: list[str],
        parameters_RIN: np.ndarray,
        parameter_names: list[str],
        on_grain: bool = False,
        njobs: int = 1,
        plot_individual_samples: bool = True,
        # do_RIN: bool = True,
        save_fig_path: str | Path | None = None,
        colors: list[str] | None = None,
        style: Style | None = None,
        put_legend_on_side: bool = False,
    ) -> None:
        seen_parameters = []
        lines = []
        fills = []
        seenSpeciesParameters = []

        if colors is None:
            colors = getColors()
        if style is None:
            style = Style()
        fig, axs = plt.subplots(
            2,
            len(species),
            sharey="row",
            sharex="col",
            figsize=(4.5, 3),
            height_ratios=(1.0, 2.0 / 3.0),
        )
        if len(species) == 1:
            axs = np.reshape(axs, (2, -1))

        for spec_idx, spec in enumerate(species):
            axs[0, spec_idx].set_title(
                convertSpeciesToLegendLabel(spec), pad=style.title_pad
            )
            self.plot_abundances_panel(
                spec,
                axs[0, spec_idx],
                on_grain=on_grain,
                njobs=njobs,
                plot_individual_samples=plot_individual_samples,
                style=style,
            )

            axs[1, spec_idx].axhline(
                0,
                color=style.zero_correlation_color,
                linestyle=style.zero_correlation_ls,
                lw=style.zero_correlation_lw,
                alpha=style.zero_correlation_alpha,
                zorder=0,
            )
            axs[1, spec_idx].fill_between(
                [0, 1e10],
                [-style.min_statistic] * 2,
                [style.min_statistic] * 2,
                color=style.weak_correlation_color,
                alpha=style.weak_correlation_alpha,
                edgecolor="none",
                zorder=0,
            )

            samples_RIN = self.get_abundances_RIN(spec, on_grain=on_grain, njobs=njobs)
            sigCorrelations = calculateSignificantCorrelations2D(
                samples_RIN,
                parameters_RIN,
                parameter_names,
                confidence_level=0.95,
                minStatistic=style.min_statistic,
            )

            if sigCorrelations is None:
                # If there are no strong enough correlations, go to next species
                continue

            # For all significant and strong correlations, plot them.
            for j, row in sigCorrelations.iterrows():
                plot_kwargs = {}
                if row["parameter"] not in seen_parameters:
                    seen_parameters.append(row["parameter"])
                    plot_kwargs["label"] = convertParameterNameToLegendLabel(
                        row["parameter"]
                    )

                paramType = get_param_type_from_param(row["parameter"])
                linestyle = get_linestyle_from_param_type(paramType)
                speciesParameter = get_species_from_param(row["parameter"])

                if speciesParameter not in seenSpeciesParameters:
                    seenSpeciesParameters.append(speciesParameter)
                colorIndex = (seenSpeciesParameters.index(speciesParameter)) % len(
                    colors
                )

                lines.append(
                    axs[1, spec_idx].plot(
                        self.times,
                        row["statistic"],
                        c=colors[colorIndex],
                        ls=linestyle,
                        **plot_kwargs,
                    )
                )

                fills.append(
                    axs[1, spec_idx].fill_between(
                        self.times,
                        np.array(row["cilow"]) - style.sampling_95_ci,
                        np.array(row["cihigh"]) + style.sampling_95_ci,
                        color=colors[colorIndex],
                        alpha=style.correlation_ci_alpha,
                        linewidth=style.correlation_ci_lw,
                        edgecolor=style.correlation_ci_edgecolor,
                    )
                )
            if not put_legend_on_side:
                leg = axs[1, spec_idx].legend(
                    handlelength=0,
                    handletextpad=0,
                    labelcolor="linecolor",
                    loc="lower left",
                    labelspacing=0.1,
                    borderaxespad=0.3,
                )
                [handle.set_visible(False) for handle in leg.legend_handles]

        for ax in axs.flat:
            ax.set_xlim(style.xlim)
            ax.set_xscale("log")
            ax.label_outer()

        axs[0, 0].set_ylim(style.ylim)
        axs[0, 0].set_yscale("log")
        axs[1, 0].set_ylim([-1.2, 1.2])
        axs[1, 0].set_yticks([-1, 0, 1])
        axs[0, 0].set_ylabel(style.abundance_label)
        axs[1, 0].set_ylabel(style.rRIN_label)

        if len(species) == 5:
            fig.supxlabel(style.time_label, x=0.525)
        elif len(species) == 4:
            fig.supxlabel(style.time_label)
        else:
            fig.supxlabel(style.time_label)

        # Set xticks
        axs[0, 0].set_xticks([1e0, 1e2, 1e4, 1e6])
        [axs[0, i].set_xticks([1e2, 1e4, 1e6]) for i in range(1, len(species))]
        [
            axs[0, i].set_xticks([1e1, 1e3, 1e5], ["", "", ""], minor=True)
            for i in range(len(species))
        ]

        plt.subplots_adjust(hspace=0, wspace=0, bottom=0.117, left=0.15, top=0.98)

        if put_legend_on_side:
            handles = [(line[0], fill) for line, fill in zip(lines, fills)]
            labels = [line[0].get_label() for line in lines]
            valid_idxs = [
                i for i, label in enumerate(labels) if not label.startswith("_")
            ]
            labels = [label for i, label in enumerate(labels) if i in valid_idxs]
            handles = [handle for i, handle in enumerate(handles) if i in valid_idxs]
            leg = axs[1, -1].legend(
                handles, labels, bbox_to_anchor=(1.0, 0.5), loc="center left"
            )

        if save_fig_path is not None:
            if not "." in save_fig_path:
                save_fig_path += ".pdf"
            plt.savefig(save_fig_path)
            plt.close()
        else:
            plt.show()

    def plot_abundances_against_parameter(
        self,
        species: list[str],
        parameter_values: np.ndarray,
        parameter_RIN: np.ndarray,
        parameter_name: str,
        on_grain: bool = False,
        njobs: int = 1,
        save_fig_path: str | Path | None = None,
        style: Style | None = None,
        xticks: tuple[float] | None = None,
    ) -> None:
        if style is None:
            style = Style()
        fig, axs = plt.subplots(
            1,
            len(species),
            sharey=True,
            sharex=True,
            figsize=(4.5, 2.598425197),
        )
        if len(species) == 1:
            axs = [axs]

        avg_param = np.average(parameter_values)
        for spec_idx, spec in enumerate(species):
            axs[spec_idx].set_title(
                convertSpeciesToLegendLabel(spec), pad=style.title_pad
            )
            samples, nominal = self.get_abundances(spec, on_grain=on_grain, njobs=njobs)
            samples_RIN = self.get_abundances_RIN(spec, on_grain=on_grain, njobs=njobs)

            allCorrelations = calculateAllCorrelations2D(
                samples_RIN,
                parameter_RIN,
                [parameter_name],
                confidence_level=0.95,
                calculateConfidenceInterval=True,
            )

            if len(self.times) > 1:
                for time_idx, time in enumerate(self.times):
                    # TODO: Fix.
                    axs[spec_idx].scatter(
                        parameter_values,
                        samples,
                        c="orange",  # TODO: Change
                        edgecolor="none",
                        marker=".",
                        s=style.abundance_marker_size,
                        alpha=style.abundance_marker_alpha,
                    )
                    axs[spec_idx].scatter(avg_param, nominal)
            else:
                log_avg = self.get_log_average_abundance(
                    spec, on_grain=on_grain, njobs=njobs
                )
                axs[spec_idx].scatter(
                    parameter_values,
                    samples,
                    c="k",
                    edgecolor="none",
                    marker=".",
                    alpha=style.abundance_marker_alpha,
                    s=style.abundance_marker_size,
                )
                axs[spec_idx].scatter(
                    avg_param,
                    nominal[self.nominal_time_indeces[0]],
                    c=style.nominal_color,
                    marker="X",
                    edgecolor=style.marker_edgecolor,
                    linewidth=style.marker_edgelinewidth,
                    s=style.nominal_marker_size,
                )
                text_rRIN = f"${{{allCorrelations['statistic'].iloc[0][0]:.2f}}}\pm{{{allCorrelations['cidiffdown'].iloc[0][0] + style.sampling_95_ci:.2f}}}$"
                axs[spec_idx].text(
                    0.5,
                    0.05,
                    text_rRIN,
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    transform=axs[spec_idx].transAxes,
                )

        axs[0].set_yscale("log")
        if "prefac" in parameter_name:
            axs[0].set_xscale("log")
        if xticks is not None:
            axs[0].set_xticks(xticks)

        axs[0].set_ylim(style.ylim)
        axs[0].set_ylabel(style.abundance_label)
        xlabel = fig.supxlabel(convertParameterNameToAxisLabel(parameter_name))
        xlabel.set_in_layout(True)

        plt.subplots_adjust(wspace=0, bottom=0.14)

        if save_fig_path is None:
            plt.show()
            plt.close()
        else:
            plt.savefig(save_fig_path)

    def plot_abundances_panel(
        self,
        spec: str,
        ax: plt.Axes,
        on_grain: bool = False,
        plot_individual_samples: bool = True,
        njobs: int = 1,
        style: Style | None = None,
    ) -> None:
        if style is None:
            style = Style()
        samples, nominal = self.get_abundances(spec, njobs=njobs, on_grain=on_grain)

        ax.plot(
            self.nominal_df["Time"],
            nominal,
            c=style.nominal_color,
            ls=style.nominal_ls,
            zorder=2.0,
        )

        log_avg = self.get_log_average_abundance(spec, on_grain=on_grain, njobs=njobs)
        if plot_individual_samples:
            nsamples = np.shape(samples)[0]
            for sample_idx in range(nsamples):
                ax.plot(
                    self.times,
                    samples[sample_idx, :],
                    c=style.sample_color,
                    lw=style.sample_lw,
                    alpha=style.sample_alpha,
                    zorder=1.99,
                )
            log_avg_color = style.average_color
        else:
            for confidence_level in [0.6827, 0.9545, 1.0]:
                cilow, cihigh = self.get_confidence_interval_of_abundances(
                    spec,
                    confidence_level=confidence_level,
                    on_grain=on_grain,
                    njobs=njobs,
                )
                ax.fill_between(
                    self.times,
                    cilow,
                    cihigh,
                    color=style.sample_color,
                    alpha=0.18,
                    edgecolor="none",
                )
            log_avg_color = style.sample_color
        ax.plot(self.times, log_avg, c=log_avg_color, ls=style.average_ls)


paramTypesLinestyle = {
    "diff": "solid",
    "diffprefac": "dashdot",
    "bind": "dashed",
    "desprefac": "dotted",
    "LH": (0, (5, 1)),
}


def get_param_type_from_param(parameter: str) -> str:
    if "LH" in parameter:
        return "LH"
    else:
        parameter_type = parameter.split()[1]
        if not parameter_type in paramTypesLinestyle:
            raise ValueError()
        return parameter_type


def get_linestyle_from_param_type(parameter_type: str) -> str:
    return paramTypesLinestyle[parameter_type]


def get_linestyle_from_param(parameter: str) -> str:
    param_type = get_parameter_type_from_parameter(parameter)
    return paramTypesLinestyle[param_type]


def get_species_from_param(parameter: str) -> str:
    param_type = get_param_type_from_param(parameter)
    if param_type == "LH":
        return convertParameterNameToLegendLabel(parameter)
    else:
        return convertSpeciesToLegendLabel(parameter.split()[0])


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


def calculateHHprefactorFromValues(
    bindingEnergy, mass, surfaceSiteDensity: float = 1.5e15
) -> float:
    return np.sqrt(
        2.0
        * surfaceSiteDensity
        * (bindingEnergy * k)
        * 1e4
        / (pi * pi * mass * atomic_mass)
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

    n = np.shape(Xranks)[0]
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
    if isinstance(dfs, list) or isinstance(dfs, np.ndarray):
        timeIndices = np.array(
            [find_nearest_idx_sorted(df["Time"].to_numpy(), time) for df in dfs]
        )
    else:
        timeIndices = np.array(find_nearest_idx_sorted(dfs["Time"].to_numpy(), time))
    return timeIndices


def getTimeFromIndices(
    dfs: list[pd.DataFrame], timeIndices: float | list[float]
) -> np.ndarray[float]:
    """Get the values of time from the array or matrix of timeIndices from every dataframe"""
    if timeIndices.ndim == 2:
        return np.array(
            [df["Time"].iloc[timeIndices[i, :]] for i, df in enumerate(dfs)]
        )
    elif timeIndices.ndim == 1:
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
        elif only_bulk:
            return df["@" + spec].iloc[timeIndex]
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
    if "sample" in filepath:
        physicalParams = (
            os.path.splitext(filepath)[0].split(os.path.sep)[-1].split("_")[:-1]
        )
    else:
        physicalParams = os.path.splitext(filepath)[0].split(os.path.sep)[-1].split("_")
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
    return f"{physicalParams[0]}K_{physicalParams[1]:.1e}cm-3_{physicalParams[2] * 1.3e-17}s-1_{physicalParams[3]}Habing"


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
    filepathsNominal: list[list[str]] | None,
    njobs: int = 1,
    format: str | None = None,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    samplesAtPhysicalParam = filepathsSamples[physicalParamIndex]
    nominalAtPhysicalParam = filepathsNominal[physicalParamIndex][0]

    sampleDFs = readOutputFiles(samplesAtPhysicalParam, njobs=njobs, format=format)
    if format is None:
        format = nominalAtPhysicalParam.split(os.path.extsep)[-1]
    elif not isinstance(format, str):
        msg = f"Format should be a string (one of ['csv', 'h5', 'hdf5']) or None, but it was type {type(format)}"
        raise TypeError(msg)

    format = format.lower()
    if format not in ["csv", "h5", "hdf5"]:
        msg = f"format should be one of ['csv', 'h5', 'hdf5'], but it was {format}"
        raise ValueError(msg)

    if format == "csv":
        nominalDF = read_output_file(nominalAtPhysicalParam)
    elif format in ["h5", "hdf5"]:
        nominalDF = read_output_file_h5(nominalAtPhysicalParam)
    else:
        raise NotImplementedError(f"Format {format} not implemented")

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
    calculate_confidence_interval: bool = True,
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
        calculateConfidenceInterval=calculate_confidence_interval,
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
            n_resamples=2000,
            method="BCa",
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
) -> tuple[list[list[str]], list[list[str]] | None]:
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
        matchStringSingleFiles = os.path.join(sampleRunsDir, f"*.{extension}")
        filepaths = glob(matchStringSingleFiles)
        if not filepaths:
            raise ValueError("No sample or nominal paths were found")
        return sortByPhysicalParamSet(filepaths), None

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
