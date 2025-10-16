import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from analysisTools import (arrheniusProb, calculateHHprefactorFromValues,
                           getColors, reactionProb, rectangularBarrierProb)

# plt.style.use(["science", "style_AA", "color_bright"])
colors = getColors()
import matplotlib as mpl


def createPlot(
    diffusionBarriers: list[float],
    reactionBarriers: list[float],
    temperature: float,
    tunnelingMass: float | list[float] = 1.0,
    barrierWidth: float = 1.4,
    prefac=1e12,
):
    diffusionRates = arrheniusProb(diffusionBarriers, temperature)

    colors = sns.color_palette("flare_r", n_colors=len(reactionBarriers))
    fig, ax = plt.subplots()

    c = np.arange(1, len(reactionBarriers) + 1)
    dummy_cax = ax.scatter(c, c, c=c, cmap=colors)  #  cmap=colors)
    ax.cla()

    if isinstance(tunnelingMass, (float, int)):
        for i, reactionBarrier in enumerate(reactionBarriers):
            reactionProbNoComp = reactionProb(
                reactionBarrier,
                temperature,
                tunnelingMass=tunnelingMass,
                barrierWidth=barrierWidth,
            )
            competitionFraction = reactionProbNoComp / (
                reactionProbNoComp + diffusionRates
            )
            reactionRate = competitionFraction * diffusionRates * prefac
            ax.plot(diffusionBarriers, reactionRate, c=colors[i])
    if isinstance(tunnelingMass, list):
        ls = ["solid", "dashed", "dotted"]
        for j, tunnelingM in enumerate(tunnelingMass):
            for i, reactionBarrier in enumerate(reactionBarriers):
                reactionProbNoComp = reactionProb(
                    reactionBarrier,
                    temperature,
                    tunnelingMass=tunnelingM,
                    barrierWidth=barrierWidth,
                )
                competitionFraction = reactionProbNoComp / (
                    reactionProbNoComp + diffusionRates
                )
                reactionRate = competitionFraction * diffusionRates * prefac
                ax.plot(diffusionBarriers, reactionRate, c=colors[i], ls=ls[j])
    else:
        raise TypeError()

    fig.colorbar(dummy_cax, ticks=c)
    plt.yscale("log")
    plt.xlabel("Diffusion barrier (K)")
    plt.ylabel("Reaction rate (s$^{-1}$)")
    plt.xlim([np.min(diffusionBarriers), np.max(diffusionBarriers)])
    plt.show()


def logFormat(string: str) -> str:
    power10 = int(np.log10(string))
    return f"$10^{{{power10}}}$"


def formatAsPower10(string: str) -> str:
    return f"$10^{{{int(string)}}}$"


def create2Dplot(
    diffusionBarriers: list[float],
    reactionBarriers: list[float],
    temperature: float,
    tunnelingMass: float | list[float] = 1.0,
    barrierWidth: float = 1.4,
    label_levels: bool = True,
    save_fig_path: str | Path | None = None,
):
    diffusionRates = arrheniusProb(diffusionBarriers, temperature)

    fig, ax = plt.subplots()
    levelsFine = np.logspace(-4, 4, num=5)
    levelsToAnnotate = [1e-4, 1e-2, 1e0, 1e2, 1e4]

    logLevelsToAnnotate = [np.log10(level) for level in levelsToAnnotate]
    cmap = sns.diverging_palette(145, 300, l=70, as_cmap=True, center="dark")
    norm = mpl.colors.BoundaryNorm([1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5], cmap.N)
    if isinstance(tunnelingMass, float) or isinstance(tunnelingMass, int):
        reactionProbs = reactionProb(
            reactionBarriers, temperature, tunnelingMass, barrierWidth=barrierWidth
        )

        DiffusionRates, ReactionProbs = np.meshgrid(diffusionRates, reactionProbs)
        DiffusionBarriers, ReactionBarriers = np.meshgrid(
            diffusionBarriers, reactionBarriers
        )

        Ratios = ReactionProbs / DiffusionRates

        # cmap = sns.color_palette("icefire", as_cmap=True)
        CS = ax.contour(
            DiffusionBarriers,
            ReactionBarriers,
            Ratios,
            levels=levelsFine,
            cmap=cmap,
            norm=norm,
        )
        ax.clabel(CS, levels=levelsToAnnotate, fmt=logFormat)
    elif isinstance(tunnelingMass, list) or isinstance(tunnelingMass, np.ndarray):
        if len(tunnelingMass) > 2:
            raise ValueError()
        ls = ["solid", "dashed"]
        for i, tunnelingM in enumerate(tunnelingMass):
            reactionProbs = reactionProb(
                reactionBarriers, temperature, tunnelingM, barrierWidth=barrierWidth
            )

            DiffusionRates, ReactionProbs = np.meshgrid(diffusionRates, reactionProbs)
            DiffusionBarriers, ReactionBarriers = np.meshgrid(
                diffusionBarriers, reactionBarriers
            )

            Ratios = ReactionProbs / DiffusionRates
            logRatios = np.log10(Ratios)

            CS = ax.contour(
                DiffusionBarriers,
                ReactionBarriers,
                Ratios,
                levels=levelsToAnnotate,
                norm=norm,
                cmap=cmap,
                linestyles=ls[i],
            )

            if label_levels:
                # https://stackoverflow.com/questions/25873681/contour-plot-labels-overlap-axes
                xmin, xmax, ymin, ymax = plt.axis()
                mid = ((xmin + xmax) / 2, (ymin + ymax) / 2)
                dx, dy = xmax - xmin, ymax - ymin
                dDim = np.array([dx, dy])

                label_pos = []
                for level, path in zip(CS.levels, CS.get_paths()):
                    if level not in levelsToAnnotate:
                        continue
                    vert = path.vertices

                    if vert.size == 0:
                        continue

                    # find point closest to center of figure
                    dist = np.linalg.norm((vert - mid) / dDim, ord=2, axis=1)
                    min_ind = np.argmin(dist)
                    label_pos.append(vert[min_ind, :])

                ax.clabel(CS, levels=levelsToAnnotate, fmt=logFormat, manual=label_pos)
        custom_leg_handles = [
            Line2D([0], [0], c="k", ls="solid"),
            Line2D([0], [0], c="k", ls="dashed"),
        ]
        custom_leg_labels = [
            f"$\mu={{{int(tunnelingM)}}}$ amu" for tunnelingM in tunnelingMass
        ]

        plt.legend(custom_leg_handles, custom_leg_labels)
    else:
        raise TypeError()
    plt.xlabel("Diffusion barrier (K)")
    plt.ylabel("Reaction barrier (K)")
    plt.xlim([np.min(diffusionBarriers), np.max(diffusionBarriers)])
    plt.ylim([np.min(reactionBarriers), np.max(reactionBarriers)])
    plt.yticks([0, 1000, 2000, 3000, 4000])

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.83, 0.11, 0.025, 0.77])
    cb = fig.colorbar(
        sm,
        boundaries=[1e-5, 1e-3, 1e-1, 1e-1, 1e3, 1e5],
        spacing="proportional",
        cax=cbar_ax,
        # ticks=levelsToAnnotate,
    )

    cb.ax.tick_params(axis="y", direction="out")
    cb.ax.minorticks_off()
    cb.ax.set_yscale("log")
    cb.ax.set_yticks(levelsToAnnotate)
    # cb.set_label(
    #     r"$P_{\mathrm{reac}}^{i+j}/\exp\left(-E_{\mathrm{diff}}^{i}/10~\mathrm{K}\right)$"
    # )
    cb.set_label(
        f"$P_{{\mathrm{{reac}}}}/\\exp\\left(-E_{{\mathrm{{diff}}}}/T\\right)$ at {int(temperature)} K"
    )

    if save_fig_path is not None:
        plt.savefig(save_fig_path)
    else:
        plt.show()


def calc_ratio_for_reaction(
    Ereac, Ediff1, Ediff2, vdiff1, vdiff2, temp, tunnelingMass=1
):
    reacRate = reactionProb(Ereac, temp, tunnelingMass) * max(vdiff1, vdiff2)
    diffuseRate = (
        arrheniusProb(Ediff1, temp) * vdiff1 + arrheniusProb(Ediff2, temp) * vdiff2
    )
    return reacRate / diffuseRate


if __name__ == "__main__":
    temp = 10.0

    diffusionBarriers = np.linspace(200, 700, num=200, endpoint=True)
    reactionBarriersFine = np.linspace(0, 4000, num=200, endpoint=True)

    if False:
        networks_dir = "/home/dijkhuis/PhD/Networks/2025-Dijkhuis-ClassicalBarrier"
        species_file = f"{networks_dir}/default_species_inertia.csv"
        reactions_file = f"{networks_dir}/default_grain_network.csv"

        species = pd.read_csv(species_file, comment="!")
        specNames = list(species["NAME"].values)

        reactions = pd.read_csv(
            reactions_file,
            names=[
                "R1",
                "R2",
                "R3",
                "P1",
                "P2",
                "P3",
                "P4",
                "Alpha",
                "Beta",
                "Gamma",
                "Unnamed1",
                "Unnamed2",
                "Unnamed3",
                "Unnamed4",
                "Unnamed5",
            ],
            comment="!",
        )

        for i, row in reactions.iterrows():
            if row["R3"] != "LH":
                continue
            # if row["R2"] != "#CO":
            #     continue
            r1_index = specNames.index(row["R1"])
            r2_index = specNames.index(row["R2"])
            r1_diffusion = species.iloc[r1_index]["BINDING ENERGY"] * 0.5
            r2_diffusion = species.iloc[r2_index]["BINDING ENERGY"] * 0.5
            min_diffusion = min(r1_diffusion, r2_diffusion)
            plt.scatter(min_diffusion, row["Gamma"])
            plt.errorbar(
                min_diffusion,
                row["Gamma"],
                min(row["Gamma"] / 2.0, 800.0),
                min(min_diffusion / 2.0, 800.0),
            )

    create2Dplot(
        diffusionBarriers,
        reactionBarriersFine,
        temp,
        tunnelingMass=[1.0, 12.0],
        label_levels=False,
    )
    # plt.axvline(325, c="gray", ls="solid", alpha=1.0, zorder=0)
    plt.savefig(f"ratioReactionDiffusion_{temp}K.pdf")
    plt.show()

    temp = 50.0

    tunnelingMass = 16.0
    Ebind1 = 4600.0
    Ediff1 = Ebind1 / 2.0
    mass1 = 17.0
    prefac1 = calculateHHprefactorFromValues(Ebind1, mass1)

    Ebind2 = 1300.0
    Ediff2 = Ebind2 / 2.0
    mass2 = 28.0
    prefac2 = calculateHHprefactorFromValues(Ebind2, mass2)

    print(prefac1)
    print(prefac2)

    EreacNominal = 1000.0
    EreacLower = 500.0
    ratio = calc_ratio_for_reaction(
        EreacNominal,
        Ediff1,
        Ediff2,
        prefac1,
        prefac2,
        temp,
        tunnelingMass=tunnelingMass,
    )
    print(ratio)
    ratio = calc_ratio_for_reaction(
        EreacLower, Ediff1, Ediff2, prefac1, prefac2, temp, tunnelingMass=tunnelingMass
    )
    print(ratio)
