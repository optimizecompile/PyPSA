"""Retrieve PyPSA example networks."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from pypsa.networks import Network
from pypsa.version import __version_semver__, __version_semver_tuple__

logger = logging.getLogger(__name__)


def _repo_url(
    master: bool = False, url: str = "https://github.com/PyPSA/PyPSA/raw/"
) -> str:
    if master or __version_semver_tuple__ < (0, 35):  # Feature was added in 0.35.0
        return f"{url}master/"
    return f"{url}v{__version_semver__}/"


def _retrieve_if_not_local(path: str | Path) -> Network:
    if not (Path.cwd() / path).exists():
        path = _repo_url() + str(path)
        Path.cwd()

    return Network(path)


def ac_dc_meshed(
    update: bool = False, from_master: bool = False, remove_link_p_set: bool = False
) -> Network:
    """
    Load the meshed AC-DC example network.

    Parameters
    ----------
    update : bool, optional
        Whether to update the locally stored network data. The default is False.
    from_master : bool, optional
        Whether to retrieve from the master branch of the pypsa repository.
    remove_link_p_set : bool, optional
        Whether to remove the link `p_set` attribute from the links.

    .. deprecated:: 0.35.0
          `from_master` and `update` are deprecated and do not have any effect.


    Returns
    -------
    pypsa.Network

    """
    if update or from_master or remove_link_p_set:
        warnings.warn(
            "The 'update' 'from_master' and 'remove_link_p_set' parameters are "
            "deprecated and do not have any effect. "
            "Example networks are always updated and retrieved for the current version."
            "Deprecated in version 0.35 and will be removed in version 1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    return _retrieve_if_not_local("examples/networks/ac-dc-meshed/ac-dc-meshed.nc")


def storage_hvdc(update: bool = False, from_master: bool = False) -> Network:
    """
    Load the storage network example of pypsa stored in the PyPSA repository.

    Parameters
    ----------
    update : bool, optional
        Whether to update the locally stored network data. The default is False.
    from_master : bool, optional
        Whether to retrieve from the master branch of the pypsa repository.

    .. deprecated:: 0.35.0
          `from_master` and `update` are deprecated and do not have any effect.

    Returns
    -------
    pypsa.Network

    """
    if update or from_master:
        warnings.warn(
            "The 'update' and 'from_master' parameters are deprecated and do not have any effect. "
            "Example networks are always updated and retrieved for the current version."
            "Deprecated in version 0.35 and will be removed in version 1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    return _retrieve_if_not_local("examples/networks/storage-hvdc/storage-hvdc.nc")


def scigrid_de(update: bool = False, from_master: bool = False) -> Network:
    """
    Load the SciGrid network example of pypsa stored in the PyPSA repository.

    Parameters
    ----------
    update : bool, optional
        Whether to update the locally stored network data. The default is False.
    from_master : bool, optional
        Whether to retrieve from the master branch of the pypsa repository.

    .. deprecated:: 0.35.0
          `from_master` and `update` are deprecated and do not have any effect.

    Returns
    -------
    pypsa.Network

    """
    if update or from_master:
        warnings.warn(
            "The 'update' and 'from_master' parameters are deprecated and do not have any effect. "
            "Example networks are always updated and retrieved for the current version."
            "Deprecated in version 0.35 and will be removed in version 1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    return _retrieve_if_not_local("examples/networks/scigrid-de/scigrid-de.nc")


def model_energy(update: bool = False, from_master: bool = False) -> Network:
    """
    Load the single-node capacity expansion model in style of model.energy.

    Parameters
    ----------
    update : bool, optional
        Whether to update the locally stored network data. The default is False.
    from_master : bool, optional
        Whether to retrieve from the master branch of the pypsa repository.

    .. deprecated:: 0.35.0
          `from_master` and `update` are deprecated and do not have any effect.

    Returns
    -------
    pypsa.Network

    """
    if update or from_master:
        warnings.warn(
            "The 'update' and 'from_master' parameters are deprecated and do not have any effect. "
            "Example networks are always updated and retrieved for the current version."
            "Deprecated in version 0.35 and will be removed in version 1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
    return _retrieve_if_not_local("examples/networks/model-energy/model-energy.nc")
