import os
import sys
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
from uclchem.makerates import io_functions
from uclchem.makerates.makerates import _get_network_from_files

from MCtools_updated import (combine_grids, create_grid,
                             extract_parameters_from_network, generate_samples,
                             get_UCLCHEM_dir, readMCparameters,
                             recompile_UCLCHEM, run_grid,
                             terminate_all_processes)

if __name__ == "__main__":
    response = input(
        "Are you sure you want to run the sensitivity analysis? Enter y if so"
    )
    if response != "y":
        print("Not running sensitivity analysis. Exiting")
        sys.exit()

    response = int(
        input(
            "Enter number of jobs. Enter -1 if you want to use all available threads."
        )
    )
    if response == -1:
        n_jobs = cpu_count()
    elif response <= cpu_count():
        n_jobs = response
    else:
        print(
            f"Set number of jobs ({response}) was higher than number of threads on PC ({cpu_count()}). Setting to number of threads"
        )
        n_jobs = cpu_count()

    response = input(
        "Do you want to simply run the calculations for the sensitivity analysis, or do a convergence test? Enter y or leave empty for sensitivity analysis"
    )
    if response == "" or response == "y":
        do_convergence = False
        print("Running calculations for sensitivity analysis")
    else:
        do_convergence = True
        print("Running calculations for convergence test")

    response = input(
        "Do you want to generate new samples, or reuse the ones set in the output directory? Enter y if you want to make new ones"
    )
    if response == "y":
        print("Generating new samples")
        generate_new_samples = True
        sys.exit()
    else:
        print("Reusing samples from file in output directory")
        generate_new_samples = False

    response = input(
        "Do you want to run the regular sensitivity analysis, or ONLY vary the reaction energy barriers, but with a wider margin? Enter nothing or y for regular, otherwise only varying reactions"
    )
    if response == "" or response == "y":
        only_reactions = False
        print("Varying everything")
    else:
        only_reactions = True
        print("Varying only reactions, with a wider standard deviation")

    if not do_convergence:
        # Number of samples to generate and run
        n_samples = 1000
        # Where to do the calculations and put the output data
        output_dir = Path(
            "/data2/dijkhuis/ChemSamplingMC/setWidthProductionTightCorrectChemdesDistributionOnlyReactions"
        )

        # Grid of physical conditions (excluding zeta=100)
        temperatures = [10.0, 20.0, 30.0, 40.0, 50.0]
        densities = [1e3, 1e4, 1e5, 1e6]
        # zetas = [0.1, 1.0, 10.0]
        # radfields = [0.1, 1.0, 10.0]
        zetas = [1.0]
        radfields = [1.0]

        # Create the grid (excluding zeta=100)
        grid_table_a = create_grid(
            ["temperature", "density", "zeta", "radfield"],
            temperatures,
            densities,
            zetas,
            radfields,
            grid_folder=output_dir,
        )

        # # Grid of physical conditions (zeta=100)
        # densities_high_zeta = [1e3, 1e4, 1e5]
        # high_zeta = [100.0]

        # # Create the grid (zeta=100)
        # grid_table_b = create_grid(
        #     ["temperature", "density", "zeta", "radfield"],
        #     temperatures,
        #     densities_high_zeta,
        #     high_zeta,
        #     radfields,
        #     grid_folder=output_dir,
        # )

        # # Create the overall grid
        # grid_table = combine_grids([grid_table_a, grid_table_b])
        grid_table = grid_table_a
    else:
        # Number of samples to test convergence
        n_samples = 2000
        # Where to do the convergence calculations and put the output data
        output_dir = Path("data_convergence")

        temperatures = [10.0, 50.0]
        densities = [1e3, 1e6]
        zetas = [1.0]
        radfields = [1.0]

        grid_table = create_grid(
            ["temperature", "density", "zeta", "radfield"],
            temperatures,
            densities,
            zetas,
            radfields,
            grid_folder=output_dir,
        )

    # Directory of this file
    working_dir = Path(os.path.dirname(__file__))

    # Directory of the file used to run the model
    run_model_file = working_dir / "run_model_updated.py"

    if not output_dir.is_dir():
        # Make output directory if it does not exist
        print(f"Making directory {output_dir}")
        output_dir.mkdir()

    # UCLCHEM directory
    uclchem_dir = get_UCLCHEM_dir()

    # Directory containing the makerates script
    makerates_dir = uclchem_dir / "Makerates"

    # Load the nominal network
    network_dir = Path("/home/dijkhuis/PhD/Networks/2025-Dijkhuis-ClassicalBarrier")
    species_file = network_dir / "default_species_inertia.csv"
    network, dropped_reactions = _get_network_from_files(
        species_file,
        [
            makerates_dir / "data/databases/umist22.csv",
            network_dir / "default_grain_network.csv",
        ],
        ["UMIST", "UCL"],
        three_phase=True,
    )

    (
        species_surf,
        lh_reacs_non_coupled,
        binding_energies_surf,
        diffusion_barriers_surf,
        diffusionprefactors_surf,
        desorptionprefactors_surf,
        barriers_non_coupled,
    ) = extract_parameters_from_network(network)

    if generate_new_samples:
        samples = generate_samples(
            network=network,
            n_samples=n_samples,
            output_path=output_dir / "MC_parameter_runs.csv",
            only_reactions=only_reactions,
        )
    else:
        # Read the parameters in the set output directory
        (
            bindingEnergies,
            diffusionBarriers,
            energyBarriers,
            diffusionPrefactors,
            desorptionPrefactors,
        ) = readMCparameters(output_dir / "MC_parameter_runs.csv")
        print(bindingEnergies)
        if (
            only_reactions
            and not diffusionBarriers.size > 0
            and not diffusionBarriers.size > 0
            and not diffusionPrefactors.size > 0
            and not desorptionPrefactors.size > 0
        ):
            bindingEnergies = np.zeros(
                shape=(len(species_surf), np.shape(energyBarriers)[1])
            )
            diffusionBarriers = np.zeros_like(bindingEnergies)
            desorptionPrefactors = np.zeros_like(bindingEnergies)
            diffusionPrefactors = np.zeros_like(bindingEnergies)
            # raise ValueError()
        samples = np.concatenate(
            [
                bindingEnergies,
                diffusionBarriers,
                diffusionPrefactors,
                desorptionPrefactors,
                energyBarriers,
            ],
            axis=0,
        ).T

    # When the network is changed, we need to call different functions for different parameters
    # (e.g. Network.change_binding_energy to change a species' binding energy)
    # so we need to know which indices are which parameters.
    bindingEnergiesStart = 0
    diffusionBarriersStart = len(binding_energies_surf)
    diffusionPrefactorsStart = diffusionBarriersStart + len(diffusion_barriers_surf)
    desorptionPrefactorsStart = diffusionPrefactorsStart + len(diffusionprefactors_surf)
    energyBarriersStart = desorptionPrefactorsStart + len(desorptionprefactors_surf)

    # Write the nominal models to the UCLCHEM files.
    os.chdir(makerates_dir)
    io_functions.write_outputs(network)
    os.chdir(working_dir)

    # Recompile UCLCHEM
    print("Recompiling UCLCHEM")
    # recompile_UCLCHEM(uclchem_dir=uclchem_dir, quiet=True)

    # List to keep track of the running processes
    processes = []

    try:
        # Run grid using the nominal model
        # processes = run_grid(
        #     grid_table,
        #     processes,
        #     run_model_file,
        #     "nominal",
        #     n_jobs,
        #     force=generate_new_samples,  # If new samples are generated, also force rerunning of models.
        # )

        for i in range(n_samples):
            if i != 773:
                continue
            # Change the corresponding binding energies and barriers in the network to the current runs values.
            if not only_reactions:
                for j, specie in enumerate(species_surf):
                    # Change binding energy
                    network.change_binding_energy(
                        specie.name, samples[i, bindingEnergiesStart + j]
                    )

                    # Change diffusion barrier
                    network.change_diffusion_barrier(
                        specie.name, samples[i, diffusionBarriersStart + j]
                    )

                    # Change diffusion prefactor
                    network.change_prefactor(
                        specie.name,
                        samples[i, diffusionPrefactorsStart + j],
                        prefactor_type="diffusion",
                    )
                    # Assume that bulk has same diffusion prefactor as surface
                    network.change_prefactor(
                        f"@{specie.name[1:]}",
                        samples[i, diffusionPrefactorsStart + j],
                        prefactor_type="diffusion",
                    )

                    # Change desorption prefactor
                    network.change_prefactor(
                        specie.name,
                        samples[i, desorptionPrefactorsStart + j],
                        prefactor_type="desorption",
                    )
                    # Assume that bulk has same desorption prefactor as surface
                    network.change_prefactor(
                        f"@{specie.name[1:]}",
                        samples[i, desorptionPrefactorsStart + j],
                        prefactor_type="desorption",
                    )

            for j, reaction in enumerate(lh_reacs_non_coupled):
                # Change reaction energy barrier
                network.change_reaction_barrier(
                    reaction, samples[i, energyBarriersStart + j]
                )

            # Write new network to files. Required as they are read at compile time
            os.chdir(makerates_dir)
            io_functions.write_outputs(network)
            os.chdir(working_dir)

            # Recompile UCLCHEM
            print("Recompiling UCLCHEM")
            recompile_UCLCHEM(uclchem_dir=uclchem_dir, quiet=True)

            # Run the grid with the changed network
            # This also keeps track of the number of running jobs.
            processes = run_grid(
                grid_table,
                processes,
                run_model_file,
                i,
                n_jobs,
                force=generate_new_samples,
            )

        # Wait until all model runs have completed to finish the python script
        for process in processes:
            process.communicate()

    except KeyboardInterrupt:
        print("Sensitivity analysis was interrupted. Terminating all processes.")
        terminate_all_processes(processes)
