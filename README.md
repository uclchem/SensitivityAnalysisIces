# A sensitivity analysis of interstellar ice chemistry in astrochemical models

**Tobias Dijkhuis[\*](#corresponding-authors), Thanja Lamberts[\*](#corresponding-authors), Serena Viti & Herma Cuppen**

DOI: .........

### Paper
You can find a PDF of the paper [here (todo)](00_paper/paper.pdf), or on arXiv [here (todo)](https://google.com).

### BibTex entry
```
@article{Dijkhuis2025,
    title = {A sensitivity analysis of interstellar ice chemistry in astrochemical models},
    author = {Dijkhuis, Tobias M. and Lamberts, Thanja and Viti, Serena and Herma, Cuppen M.},
    year = {2025},
    doi = {},
    url = {},
    month = {},
    journal = {Astronomy \& Astrophysics}, % Replace by \aap if possible
    publisher = {EDP Sciences},
    volume = {},
    number = {},
    issue = {}
    pages = {},
}
```

## Replicating results
All the data shown in the paper (and much more) can be generated using the scripts provided in this repository.

### Environment setup
I would recommend creating a new python environment, to make sure that you have all of the required packages,
and none of the packages conflict with the ones installed on your system before.

The environment that was used for calculations and data analysis can be copied and activated by doing

    conda env create -f environment.yml
    conda activate sensitivity_analysis_ices

Then, install UCLCHEM into this new environment

    cd model
    python3 -m pip install -e .

This will take some time, because UCLCHEM will be need to be compiled.

> **Note:** This is a custom version of UCLCHEM, so you need to use the one provided in this repository.

### Data generation
The data for the standard cosmic ray ionization rate ($\zeta=1.3\times10^{-17}$ s$^{-1}$) and UV field strength 
($F_{\mathrm{UV}}=1$ Habing) can be obtained from the [Zenodo repository](https://doi.org/10.5281/zenodo.17463693).
After downloading this data and putting it in the [02_data](02_data) directory and extracting the contents you should be good to go.
The contents can be extracted by doing (on Ubuntu at least, check for your specific operating system)

    unzip data_sensitivity_analysis_ices.zip 

This should result in two directories, `varying_all` and `varying_reactions`. The first varies all parameters,
and the second varies only the reaction energy barriers, but with a wider distribution (used for Fig. D3 in the paper).

If desired, the data for other values of the cosmic ray ionization rate and UV field strengths
can be generated as such.

    cd 01_data_generation
    python3 run_sensitivity_analysis.py

> **Note:** You need to specify the desired destination directory manually in `run_sensitivity_analysis.py`.

### Data analysis
Once the data has been generated, it can be analyzed using tools in the [data analysis directory](03_data_analysis).
In [create_figures.py](03_data_analysis/create_figures.py), there is code to create all the figures
shown in the paper. 

    cd 03_data_analysis
    python3 create_figures.py

You can also use this code as inspiration for different analyses, like for different species.

### Figures
All the created figures will be placed in the [figures directory](04_figures).
This already contains all figures used in the paper.

Additional figures, not shown in the paper, can be created by doing 

    cd 03_data_analysis
    python3 create_additional_figures.py

This creates a bunch of additional figures, similar to [Figure 2](04_figures/fig2.pdf) but for
different physical conditions. They are shown in the directory [05_additional_figures](05_additional_figures).

## Corresponding authors
 - Tobias Dijkhuis: <t.m.dijkhuis@lic.leidenuniv.nl>
 - Thanja Lamberts: <a.l.m.lamberts@lic.leidenuniv.nl>

## ORCID
 - Tobias Dijkhuis: <https://orcid.org/0009-0009-2498-6429>
 - Thanja Lamberts: <https://orcid.org/0000-0001-6705-2022>
 - Serena Viti:     <https://orcid.org/0000-0001-8504-8844>
 - Herma Cuppen:    <https://orcid.org/0000-0003-4397-0739>


