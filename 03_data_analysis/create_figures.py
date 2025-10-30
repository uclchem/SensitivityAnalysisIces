import os
import sys

import numpy as np

from pathlib import Path

from analysisTools import DataManager, Style, set_environ_njobs, set_rc_params
from competition import create2Dplot


def create_figure_1(manager: DataManager, output_dir: Path | str = ".") -> None:
    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    T = 10.0
    nH = 1e5
    zeta = 1.0
    radfield = 1.0

    path = Path(output_dir) / "fig1.pdf"
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
        xticks=[250.0, 450.0],
    )
    print(f"Created figure 1 at {path}")


def create_figure_2(manager: DataManager, output_dir: Path | str = ".") -> None:
    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    T = 10.0
    nH = 1e5
    zeta = 1.0
    radfield = 1.0
    path = Path(output_dir) / "fig2.pdf"
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
            "#bd1f01",  # N, Color of N in Fig 3.
            "#3f90da",  # H Color of H in Fig 3.
            "#e76300",  # CH3, Color of CH3 in Fig 3.
            "#94a4a2",  # O, Color of O in Fig 3.
            "#832db6",  # NH, not in Fig. 3.
            "#ffa90e",  # CH, Color of CH in Fig 3.
            "#b9ac70",
            "#a96b59",
            "#717581",
            "#92dadd",
            "#222222",
        ],  # https://arxiv.org/pdf/2107.02270
        put_legend_on_side=True,
        plot_individual_samples=True,
    )
    print(f"Created figure 2 at {path}")


def create_figure_3(manager: DataManager, output_dir: Path | str = ".") -> None:
    # Create BIG plot for H2O, CO, CO2, CH3OH, NH3
    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    temps = [10.0, 30.0, 50.0]
    densities = [1e3, 1e6]
    zeta = 1.0
    radfield = 1.0

    path = Path(output_dir) / "fig3.pdf"
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


def create_figure_4(manager: DataManager, output_dir: Path | str = ".") -> None:
    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    temps = [10.0, 20.0, 30.0, 40.0, 50.0]
    densities = [1e3, 1e4, 1e5, 1e6]
    zeta = 1.0
    radfield = 1.0
    confidence_level = 0.6827
    path = Path(output_dir) / "fig4.pdf"
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


def create_figure_5(output_dir: Path | str = ".") -> None:
    path = Path(output_dir) / "fig5.pdf"
    temp = 10.0

    diffusionBarriers = np.linspace(200, 700, num=200, endpoint=True)
    reactionBarriers = np.linspace(0, 4000, num=200, endpoint=True)

    create2Dplot(
        diffusionBarriers,
        reactionBarriers,
        temp,
        tunnelingMass=[1.0, 12.0],
        label_levels=False,
        save_fig_path=path,
    )
    print(f"Created figure 5 at {path}")


def create_figure_D1(manager: DataManager, output_dir: Path | str = ".") -> None:
    species = ["CH3CHO", "HCOOCH3", "CH3OCH3", "NH2CHO"]
    temps = [10.0, 30.0, 50.0]
    densities = [1e3, 1e6]
    zeta = 1.0
    radfield = 1.0

    path = Path(output_dir) / "figD1.pdf"
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


def create_figure_D2(manager: DataManager, output_dir: Path | str = ".") -> None:
    species = ["CH3CHO", "HCOOCH3", "CH3OCH3", "NH2CHO"]
    temps = [10.0, 20.0, 30.0, 40.0, 50.0]
    densities = [1e3, 1e4, 1e5, 1e6]
    zeta = 1.0
    radfield = 1.0
    confidence_level = 0.6827
    path = Path(output_dir) / "figD2.pdf"
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
    print(f"Created figure D2 at {path}")


def create_figure_D3(manager: DataManager, output_dir: Path | str = "."):
    species = ["H2O", "CO", "CO2", "CH3OH", "NH3"]
    temps = [10.0, 20.0, 30.0, 40.0, 50.0]
    densities = [1e3, 1e4, 1e5, 1e6]
    zeta = 1.0
    radfield = 1.0
    confidence_level = 0.6827
    path = Path(output_dir) / "figD3.pdf"
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
    print(f"Created figure D3 at {path}")


if __name__ == "__main__":
    set_rc_params()
    njobs = 20
    set_environ_njobs(njobs)

    output_dir = Path("../04_figures")

    all_directory = Path("../02_data/varying_all")
    all_manager = DataManager(all_directory, njobs=njobs)

    create_figure_1(all_manager, output_dir=output_dir)
    create_figure_2(all_manager, output_dir=output_dir)
    create_figure_3(all_manager, output_dir=output_dir)
    create_figure_4(all_manager, output_dir=output_dir)
    create_figure_5(output_dir=output_dir)
    create_figure_D1(all_manager, output_dir=output_dir)
    create_figure_D2(all_manager, output_dir=output_dir)

    reactions_directory = Path("../02_data/varying_reactions")
    reactions_manager = DataManager(reactions_directory, njobs=njobs)
    create_figure_D3(reactions_manager, output_dir=output_dir)
