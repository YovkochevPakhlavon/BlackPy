"""Curated, citable registry of observational data.

>>> import blackpy as bp
>>> gro = bp.data.load("qpo/gro_j1655_40")
>>> datasets = bp.data.list()        # prints the registry with verification status

Every value in the registry was read from the cited source, with the exact
location recorded; values that could not be verified are ``null``. All
entries carry ``verified_by_human: false`` until the project owner has
checked them.
"""

from blackpy.data.registry import available, load, load_file, parse_dataset, registry_path
from blackpy.data.registry import list_datasets as list  # noqa: A004
from blackpy.data.schema import (
    Dataset,
    MassPrior,
    Measurement,
    QPODataset,
    Reference,
    RegistryError,
    ShadowDataset,
    SpinMeasurement,
)

__all__ = [
    "Dataset",
    "MassPrior",
    "Measurement",
    "QPODataset",
    "Reference",
    "RegistryError",
    "ShadowDataset",
    "SpinMeasurement",
    "available",
    "list",
    "load",
    "load_file",
    "parse_dataset",
    "registry_path",
]
