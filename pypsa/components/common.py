"""General utility functions for PyPSA."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypsa.components.components import Components
from pypsa.deprecations import COMPONENT_ALIAS_DICT

if TYPE_CHECKING:
    from pypsa.typing import NetworkType


def as_components(n: NetworkType, value: str | Components) -> Components:
    """Get component instance from string.

    E.g. pass 'Generator', 'generators' or Components class instance to get the
    corresponding Components class instance.

    Parameters
    ----------
    value : str | Components
        String or Components class instance.
    n : pypsa.Network
        Network instance to which the components are attached.

    Returns
    -------
    Components
        Components class instance.

    """
    if isinstance(value, str):
        if value in COMPONENT_ALIAS_DICT:
            value = COMPONENT_ALIAS_DICT[value]
        return getattr(n.components, value)
    if isinstance(value, Components):
        if value.n is None:
            msg = "Passed component must be attached to the same network."
            raise ValueError(msg)
        if value.n is not n:
            msg = "Passed component is attached to a different network."
            raise ValueError(msg)
        return value
    msg = "Value must be a string or Components class instance."
    raise TypeError(msg)
