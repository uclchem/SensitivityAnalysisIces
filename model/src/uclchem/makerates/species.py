import logging
from collections import Counter

import numpy as np

speciesNfields = 14

elementList = [
    "H",
    "D",
    "HE",
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "CL",
    "LI",
    "NA",
    "MG",
    "SI",
    "PAH",
    "15N",
    "13C",
    "18O",
    "E-",
    "FE",
]
elementMass = [
    1,
    2,
    4,
    12,
    14,
    16,
    19,
    31,
    32,
    35,
    3,
    23,
    24,
    28,
    420,
    15,
    13,
    18,
    0,
    56,
]
symbols = ["#", "@", "*", "+", "-", "(", ")"]


def is_number(s) -> bool:
    """Try to convert input to a float, if it succeeds, return True.

    Args:
        s: Input element to check for

    Returns:
        bool: True if a number, False if not.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


class Species:
    """Species is a class that holds all the information about an individual species in the
    network. It also has convenience functions to check whether the species is a gas or grain
    species and to help compare between species.
    """

    def __init__(self, inputRow, do_checks=True):
        """A class representing chemical species, it reads in rows which are formatted as follows:
        NAME,MASS,BINDING ENERGY,DESORPTION PREFACTOR,DIFFUSION BARRIER, DIFFUSION PREFACTOR,
        SOLID FRACTION,MONO FRACTION,VOLCANO FRACTION,ENTHALPY,Ix,Iy,Iz,SYMMETRY NUMBER
        Args:
            inputRow (list):
        """
        self.name = inputRow[0].upper()
        self.mass = int(inputRow[1])

        self.is_refractory = str(inputRow[2]).lower() == "inf"
        if self.is_refractory:
            self.binding_energy = 99.9e9
        else:
            self.binding_energy = float(inputRow[2])
        self.vdes = float(inputRow[3])

        self.diffusion_barrier = float(inputRow[4])
        self.vdiff = float(inputRow[5])

        self.solidFraction = float(inputRow[6])
        self.monoFraction = float(inputRow[7])
        self.volcFraction = float(inputRow[8])
        self.enthalpy = float(inputRow[9])

        if do_checks and self.is_grain_species() and self.enthalpy == 0:
            if (
                self.name[1:] in ["BR2", "I2", "N2", "CL2", "H2", "O2", "F2"]
                or self.name[1:] in ["AL", "MG", "HE"]
                or len(self.name[1:]) == 1
            ):
                pass
            else:
                logging.warning(
                    f"Grain species {self.name} was not given a formation enthalpy. This will result in incorrect ChemDes rates"
                )

        try:
            self.Ix = float(inputRow[10])
            self.Iy = float(inputRow[11])
            self.Iz = float(inputRow[12])
            self.symmetry_factor = int(inputRow[13])
        except ValueError:
            self.Ix = -999.0
            self.Iy = -999.0
            self.Iz = -999.0
            self.symmetry_factor = -1

        self.n_atoms = 0

        # in first instance, assume species freeze/desorb unchanged
        # this is updated by `check_freeze_desorbs()` later.
        if self.is_grain_species():
            # this will make any excited species desorb as their base counterparts
            if "*" in self.name:
                self.desorb_products = [self.name[1:-1], "NAN", "NAN", "NAN"]
            else:
                self.desorb_products = [self.name[1:], "NAN", "NAN", "NAN"]
        else:
            self.freeze_products = {}

        if do_checks:
            self.find_constituents()

        if (
            do_checks
            and self.is_grain_species()
            and not self.name in ["SURFACE", "BULK"]
        ):
            self.check_symmetry_factor()

    def get_name(self) -> str:
        """Get the name of the chemical species.

        Returns:
            str: The name
        """
        return self.name

    def get_mass(self) -> int:
        """Get the molecular mass of the chemical species

        Returns:
            int: The molecular mass
        """
        return self.mass

    def set_desorb_products(self, new_desorbs: list[str]) -> None:
        """Set the desorption products for species on the surface or in the bulk.
        It is assumed that there is only one desorption pathway.

        Args:
            new_desorbs (list[str]): The new desorption products
        """
        self.desorb_products = new_desorbs

    def get_desorb_products(self) -> list[str]:
        """Obtain the desorbtion products of ice species

        Returns:
            list[str]: The desorption products
        """
        return self.desorb_products

    def set_freeze_products(self, product_list: list[str], freeze_alpha: float) -> None:
        """Add the freeze products of the species, one species can have several freeze products.

        Args:
            product_list (list[str]): The list of freeze out products
            freeze_alpha (float): The freeze out ratio.

        It is called alpha, since it is derived from the alpha column in the UCLCHEM reaction format:
        https://github.com/uclchem/UCLCHEM/blob/08d37f8c3063f8ff8a9a7aa16d9eff0ed4f99538/Makerates/src/network.py#L160
        """

        self.freeze_products[",".join(product_list)] = freeze_alpha

    def get_freeze_products(self) -> dict[list[str], float]:
        """Obtain the product to which the species freeze out

        Returns:
            dict[str, float]: Reactions and their respective freeze out ratios.

        Yields:
            Iterator[dict[str, float]]: Iterator that returns all of the freeze out reactions with ratios
        """
        keys = self.freeze_products.keys()
        values = self.freeze_products.values()
        logging.debug(f"freeze keys: {keys}, products {values}")
        for key, value in zip(keys, values):
            yield key.split(","), value

    def get_freeze_products_list(self) -> list[list[str]]:
        """Returns all the freeze products without their ratios

        Returns:
            list[list[str]]: List of freeze products
        """
        # TODO: Write an unit test for get_freeze_product_behaivour
        return [key.split(",") for key in self.freeze_products.keys()]

    def get_freeze_alpha(self, product_list: list[str]) -> float:
        """Obtain the freeze out ratio of a species for a certain reaction

        Args:
            product_list (list[str]): For a specific reaction, get the freezeout ratio

        Returns:
            float: The freezeout ratio
        """
        return self.freeze_products[",".join(product_list)]

    def is_grain_species(self) -> bool:
        """Return whether the species is a species on the grain

        Returns:
            bool: True if it is a grain species.
        """
        return (
            self.name in ["BULK", "SURFACE"]
            or self.name.startswith(
                "#",
            )
            or self.name.startswith("@")
        )

    def is_surface_species(self) -> bool:
        """Checks if the species is on the surface

        Returns:
            bool: True if a surface species
        """
        return self.name.startswith("#")

    def is_bulk_species(self) -> bool:
        """Checks if the species is in the bulk

        Returns:
            bool: True if a bulk species
        """
        return self.name.startswith("@")

    def is_ion(self) -> bool:
        """Checks if the species is ionized, either postively or negatively.

        Returns:
            bool: True if it is an ionized
        """
        return self.name.endswith("+") or self.name.endswith("-")

    def add_default_freeze(self) -> None:
        """Adds a defalt freezeout, which is freezing out to the species itself, but with no ionization."""
        freeze = "#" + self.name
        if freeze[-1] in ["+", "-"]:
            freeze = freeze[:-1]
        if self.name == "E-":
            freeze = ""
        self.set_freeze_products([freeze, "NAN", "NAN", "NAN"], 1.0)

    def find_constituents(self, quiet=False):
        """Loop through the species' name and work out what its consituent
        atoms are. Then calculate mass and alert user if it doesn't match
        input mass.
        """
        if self.name in ["SURFACE", "BULK", "E-"]:
            return
        speciesName = self.name[:]
        i = 0
        atoms = []
        bracket = False
        bracketContent = []
        # loop over characters in species name to work out what it is made of
        while i < len(speciesName):
            # if character isn't a #,+ or - then check it otherwise move on
            if speciesName[i] not in symbols:
                if i + 1 < len(speciesName):
                    # if next two characters are (eg) 'MG' then atom is Mg not M and G
                    if speciesName[i : i + 3] in elementList:
                        j = i + 3
                    elif speciesName[i : i + 2] in elementList:
                        j = i + 2
                    # otherwise work out which element it is
                    elif speciesName[i] in elementList:
                        j = i + 1

                # if there aren't two characters left just try next one
                elif speciesName[i] in elementList:
                    j = i + 1
                # if we've found a new element check for numbers otherwise print error
                if j > i:
                    if bracket:
                        bracketContent.append(speciesName[i:j])
                    else:
                        atoms.append(speciesName[i:j])  # add element to list
                    if j < len(speciesName):
                        if is_number(speciesName[j]):
                            if int(speciesName[j]) > 1:
                                for k in range(1, int(speciesName[j])):
                                    if bracket:
                                        bracketContent.append(speciesName[i:j])
                                    else:
                                        atoms.append(speciesName[i:j])
                                i = j + 1
                            else:
                                i = j
                        else:
                            i = j
                    else:
                        i = j
                else:
                    raise ValueError(
                        f"Contains elements not in element list: {speciesName}"
                    )
                    logging.warning(speciesName[i])
                    logging.warning(
                        "\t{0} contains elements not in element list:".format(
                            speciesName
                        )
                    )
                    logging.warning(elementList)
            else:
                # if symbol is start of a bracketed part of molecule, keep track
                if speciesName[i] == "(":
                    bracket = True
                    bracketContent = []
                    i += 1
                # if it's the end then add bracket contents to list
                elif speciesName[i] == ")":
                    if is_number(speciesName[i + 1]):
                        for k in range(0, int(speciesName[i + 1])):
                            atoms.extend(bracketContent)
                        i += 2
                    else:
                        atoms.extend(bracketContent)
                        i += 1
                # otherwise move on
                else:
                    i += 1

        self.n_atoms = len(atoms)
        mass = 0
        for atom in atoms:
            mass += elementMass[elementList.index(atom)]
        if mass != int(self.mass):
            self.mass = int(mass)
            if not quiet:
                logging.warning(
                    f"Input mass of {self.name} ({self.mass}) does not match calculated mass of constituents, using calculated mass: {int(mass)}"
                )
        counter = Counter()
        for element in elementList:
            if element in atoms:
                counter[element] = atoms.count(element)
        return counter

    def calculate_rotational_partition_factor(self) -> float:
        """Calculate 1/sigma*(SQRT(IxIyIz)) for non-linear molecules, and
        1/sigma*(SQRT(IyIz)) for linear molecules"""
        if self.n_atoms == 1:
            # For atoms, this is undefined, just return a value such that
            # it is clearly an atomic species.
            return -1.0
        if self.Ix == self.Iy == self.Iz == -999.0:
            # For species without custom input Ix, Iy and Iz, we cannot do this,
            # so we will use an empirical formula instead.
            return -999
        # Ix, Iy and Iz are in units of amu Angstrom^2,
        # so need to convert to kg m2
        amu = 1.66053907e-27  # kg/amu
        scalingFactor = 1e50
        if not self.is_linear():
            return (
                (1.0 / self.symmetry_factor)
                * np.sqrt(self.Ix * self.Iy * self.Iz * amu**3 / 1e60)
                * scalingFactor
            )
        else:
            return (
                (1.0 / self.symmetry_factor)
                * np.sqrt(self.Iy * self.Iz * amu**2 / 1e40)
                * scalingFactor
            )

    def is_linear(self) -> bool:
        if self.n_atoms == 1:
            # Atomic species are not linear. Does not really matter, this will be
            # filtered out anyway by UCLCHEM when vdes is calculated.
            return False
        if self.n_atoms == 2:
            return True
        if self.Ix == 0 and self.Iy == 0 and self.Iz == 0:
            raise NotImplemented(
                "is_linear is not implemented for molecules without custom input Ix, Iy and Iz"
            )
        if not self.is_grain_species():
            raise NotImplemented("is_linear is not implemented for gas-phase species")
        return self.Ix == 0

    def check_symmetry_factor(self) -> None:
        if self.n_atoms == 1:  # Nothing to check
            return
        if self.n_atoms > 2:  # Can not correctly check everything
            return
        constituents = self.find_constituents(quiet=True)
        if (
            len(constituents) == 2
        ):  # Only one constituent, i.e. both atoms are the same element.
            if self.symmetry_factor == 1:
                return
            msg = f"For diatomic molecule consisting of two different atoms (in this case {self.name}), the symmetry factor should be 1, but was given to be {self.symmetry_factor}. Correcting to 1."
            logging.warning(msg)
            self.symmetry_factor = 1
            return
        if self.symmetry_factor == 2:
            return
        msg = f"For diatomic molecule consisting of two of the same atoms (in this case {self.name}), the symmetry factor should be 2, but was given to be {self.symmetry_factor}. Correcting to 2."
        logging.warning(msg)
        self.symmetry_factor = 2

    def get_n_atoms(self) -> int:
        """Obtain the number of atoms in the molecule

        Returns:
            int: The number of atoms
        """
        return self.n_atoms

    def set_n_atoms(self, new_n_atoms: int) -> None:
        """Set the number of atoms

        Args:
            new_n_atoms (int): The new number of atoms
        """
        self.n_atoms = new_n_atoms

    def __eq__(self, other):
        """Check for equality based on either a string or another Species instance.

        Args:
            other (str, Species): Another species

        Raises:
            NotImplementedError: We can only compare between species or strings of species.

        Returns:
            bool: True if two species are identical.
        """
        if isinstance(other, Species):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            raise NotImplementedError(
                "We can only compare between species or strings of species"
            )

    def __lt__(self, other) -> bool:
        """Compare the mass of the species

        Args:
            other (Species): Another species instance

        Returns:
            bool: True if less than the other species
        """
        return self.mass < other.mass

    def __gt__(self, other) -> bool:
        """Compare the mass of the species

        Args:
            other (Species): Another species instance

        Returns:
            bool: True if larger than than the other species
        """
        return self.mass > other.mass

    def __repr__(self) -> str:
        return f"Specie: {self.name}"

    def __str__(self) -> str:
        return self.name
