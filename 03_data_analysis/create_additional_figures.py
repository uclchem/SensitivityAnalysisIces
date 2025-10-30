import numpy as np

from pathlib import Path

from analysisTools import DataManager, Style, set_environ_njobs, set_rc_params

if __name__ == "__main__":
    set_rc_params()
    njobs = 20
    set_environ_njobs(njobs)

    output_dir = Path("../05_additional_figures")

    all_directory = Path("../02_data/varying_all")
    all_manager = DataManager(all_directory, njobs=njobs)

    all_manager.create_standard_plots_for_all_physical_conditions(
        species=["H2O", "CO", "CO2", "CH3OH", "NH3"],
        on_grain=True,
        times=np.logspace(0, 6, num=100),
        filter_zeta=1.0,
        filter_radfield=1.0,
        plot_individual_samples=True,
        put_legend_on_side=True,
    )
