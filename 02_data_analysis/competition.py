import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from analysisTools import arrheniusProb, getColors, reactionProb, rectangularBarrierProb

plt.style.use(["science", "font_cmubright", "color_bright"])
colors = getColors()


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


def create2Dplot(
    diffusionBarriers: list[float],
    reactionBarriers: list[float],
    temperature: float,
    tunnelingMass: float | list[float] = 1.0,
    barrierWidth: float = 1.4,
):
    diffusionRates = arrheniusProb(diffusionBarriers, temperature)

    fig, ax = plt.subplots()
    levelsFine = np.logspace(-4, 4, num=5)
    levelsToAnnotate = [1e-4, 1e-2, 1e0, 1e2, 1e4]
    cmap = sns.diverging_palette(145, 300, l=70, as_cmap=True, center="dark")
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
            norm="log",
            cmap=cmap,
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

            CS = ax.contour(
                DiffusionBarriers,
                ReactionBarriers,
                Ratios,
                levels=levelsFine,
                norm="log",
                cmap=cmap,
                linestyles=ls[i],
            )

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
    return


if __name__ == "__main__":
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

    diffusionBarriers = np.linspace(200, 700, num=200, endpoint=True)
    temp = 10.0

    reactionBarriersFine = np.linspace(0, 5000, num=200, endpoint=True)
    create2Dplot(
        diffusionBarriers, reactionBarriersFine, temp, tunnelingMass=[1.0, 5.0]
    )
    plt.axvline(325, c="gray", ls="solid", alpha=1.0, zorder=0)
    # for i, row in reactions.iterrows():
    #     if row["R3"] != "LH":
    #         continue
    #     if row["R2"] != "#CO":
    #         continue
    #     r1_index = specNames.index(row["R1"])
    #     r2_index = specNames.index(row["R2"])
    #     r1_diffusion = species.iloc[r1_index]["BINDING ENERGY"] * 0.5
    #     r2_diffusion = species.iloc[r2_index]["BINDING ENERGY"] * 0.5
    #     min_diffusion = min(r1_diffusion, r2_diffusion)
    #     plt.scatter(min_diffusion, row["Gamma"])
    #     plt.errorbar(
    #         min_diffusion,
    #         row["Gamma"],
    #         min(row["Gamma"] / 2.0, 800.0),
    #         min(min_diffusion / 2.0, 800.0),
    #     )
    plt.savefig(f"ratioReactionDiffusion_{temp}K.pdf")
    plt.show()
