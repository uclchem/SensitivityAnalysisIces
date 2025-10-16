import os

import numpy as np

from analysisTools import DataManager, Style, set_environ_njobs, set_rc_params
from competition import create2Dplot


def create_figure_1(manager: DataManager) -> None:
    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    T = 10.0
    nH = 1e5
    zeta = 1.0
    radfield = 1.0

    path = "fig1.pdf"
    manager.plot_abundances_against_parameter(
        T,
        nH,
        zeta,
        radfield,
        species,
        "#H diff",
        1e4,
        on_grain=True,
        save_fig_path=path,
    )
    print(f"Created figure 1 at {path}")


def create_figure_2(manager: DataManager) -> None:
    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    T = 10.0
    nH = 1e5
    zeta = 1.0
    radfield = 1.0
    path = f"fig2.pdf"
    manager.create_standard_plot(
        T,
        nH,
        zeta,
        radfield,
        species,
        times=np.logspace(0, 6, 100),
        on_grain=True,
        save_fig_path=path,
        colors=[
            "#bd1f01",  # Color of N in Fig 3.
            "#3f90da",  # Color of H in Fig 3.
            "#ffa90e",  # Color of CH in Fig 3.
            "#a96b59",  # Color of CH3 in Fig 3.
            "#94a4a2",  # Color of O in Fig 3.
            "#832db6",
            "#e76300",
            "#b9ac70",
            "#717581",
            "#92dadd",
            "#222222",
        ],  # https://arxiv.org/pdf/2107.02270
        put_legend_on_side=True,
        plot_individual_samples=True,
    )
    print(f"Created figure 2 at {path}")


def create_figure_3(manager: DataManager) -> None:
    # Create BIG plot for H2O, CO, CO2, CH3OH, NH3
    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    temps = [10.0, 30.0, 50.0]
    densities = [1e3, 1e6]
    zeta = 1.0
    radfield = 1.0

    path = "fig3.pdf"
    manager.create_big_plot(
        temps,
        densities,
        zeta,
        radfield,
        species,
        on_grain=True,
        times=np.logspace(0, 6, num=100),
        save_fig_path=path,
        colors=[
            "#3f90da",
            "#ffa90e",
            "#bd1f01",
            "#94a4a2",
            "#832db6",
            "#a96b59",
            "#e76300",
            "#b9ac70",
            "#717581",
            "#92dadd",
            "#222222",
        ],  # Color palette from https://arxiv.org/pdf/2107.02270, with black added.
    )
    print(f"Created figure 3 at {path}")


def create_figure_4(manager: DataManager) -> None:
    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    temps = [10.0, 20.0, 30.0, 40.0, 50.0]
    densities = [1e3, 1e4, 1e5, 1e6]
    zeta = 1.0
    radfield = 1.0
    confidence_level = 0.6827
    path = "fig4.pdf"
    manager.create_widths_plot(
        temps,
        densities,
        zeta,
        radfield,
        species,
        confidence_level=confidence_level,
        on_grain=True,
        times=np.logspace(0, 6, num=100),
        save_fig_path=path,
    )
    print(f"Created figure 4 at {path}")


def create_figure_5() -> None:
    path = "fig5.pdf"
    temp = 10.0

    diffusionBarriers = np.linspace(200, 700, num=200, endpoint=True)
    reactionBarriers = np.linspace(0, 4000, num=200, endpoint=True)

    create2Dplot(
        diffusionBarriers,
        reactionBarriers,
        temp,
        tunnelingMass=[1.0, 6.0],
        label_levels=False,
        save_fig_path=path,
    )
    print(f"Created figure 5 at {path}")


def create_figure_D1(manager: DataManager) -> None:
    species = ["CH3CHO", "HCOOCH3", "CH3OCH3", "NH2CHO"]
    temps = [10.0, 30.0, 50.0]
    densities = [1e3, 1e6]
    zeta = 1.0
    radfield = 1.0

    path = "figD1.pdf"
    manager.create_big_plot(
        temps,
        densities,
        zeta,
        radfield,
        species,
        on_grain=True,
        times=np.logspace(0, 6, num=100),
        save_fig_path=path,
        colors=[
            "#5790fc",
            "#f89c20",
            "#e42536",
            "#964a8b",
            "#9c9ca1",
            "#7a21dd",
        ],  # https://arxiv.org/pdf/2107.02270
    )
    print(f"Created figure D1 at {path}")


def create_figure_D2(manager: DataManager) -> None:
    species = ["CH3CHO", "HCOOCH3", "CH3OCH3", "NH2CHO"]
    temps = [10.0, 20.0, 30.0, 40.0, 50.0]
    densities = [1e3, 1e4, 1e5, 1e6]
    zeta = 1.0
    radfield = 1.0
    confidence_level = 0.6827
    path = "figD2.pdf"
    manager.create_widths_plot(
        temps,
        densities,
        zeta,
        radfield,
        species,
        confidence_level=confidence_level,
        on_grain=True,
        times=np.logspace(0, 6, num=10),
        save_fig_path=path,
    )
    print(f"Created figure D2 at {path}")


if __name__ == "__main__":
    set_rc_params()
    njobs = 20
    set_environ_njobs(njobs)

    style = Style()

    style.nominal_color = "#FFB8F2"
    style.average_color = "#8EFF8B"

    calculation_directory = "/data2/dijkhuis/ChemSamplingMC/setWidthProductionTight"
    manager = DataManager(calculation_directory, style=style, njobs=njobs)

    create_figure_1(manager)
    create_figure_2(manager)
    create_figure_3(manager)
    create_figure_4(manager)
    create_figure_5()
    create_figure_D1(manager)
    create_figure_D2(manager)
