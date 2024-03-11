"""Module providing tooling to elaborate a dictionary object."""
from typing import Dict, Set, Any


def dict_of_dict(D: Dict) -> bool:
    """Return whether given dictionary contains only dictionaries.

    Parameters
    -------------------------------
    D: Dict
        Dictionary of which to determine if contains only dictionaries.

    Returns
    -------------------------------
    Boolean representing whether given dictionary contains only dictionaries.
    """
    return all(isinstance(d, dict) for d in D.values())


def min_depth(D: Dict) -> int:
    """Return minimum depth of given dictionary.

    Parameters
    -------------------------------
    D: Dict
        Dictionary of which to determine the width

    Returns
    -------------------------------
    Minimum depth of dictionary.
    """
    return 1 + min(
        min_depth(d) for d in D.values()
    ) if dict_of_dict(D) else 0


def axis_keys(D: Dict, axis: int) -> Set[Any]:
    """Return set of keys at given axis.

    Parameters
    -------------------------------
    D: Dict
        Dictionary to determine keys of.
    Axis:int
        Depth of keys.

    Returns
    -------------------------------
    The set of keys at given axis
    """
    return set.union(*[
        axis_keys(d, axis-1) for d in D.values()
    ]) if axis else set(D.keys())


def reindex_key(D: Dict, key: Any, axis: int) -> Dict:
    """Return reindex dictionary to given key.

    Parameters
    -------------------------------
    D: Dict
        Dictionary to reindex.
    key: Any
        Key to reindex.
    axis: int
        Depth of key to reindex.

    Returns
    -------------------------------
    Reindexed dictionary.
    """
    if axis == 0:
        return D.get(key)
    return {
        subkey: value
        for subkey, value in
        (
            (
                subkey,
                value
                if not isinstance(value, dict)
                else
                reindex_key(
                    value,
                    key,
                    axis-1
                )
            )
            for subkey, value in D.items()
        )
        if value is not None and (not isinstance(value, dict) or len(value) != 0)
    }


def transpose_dict(D: Dict, axis: int) -> Dict:
    """Transpose given dictionary on given axis.

    Parameters
    -------------------------------
    D: Dict
        Dictionary to transpose.
    axis: int
        Axis to traspose.

    Returns
    -------------------------------
    Transposed dictionary.
    """
    return {
        key: reindex_key(D, key, axis)
        for key in axis_keys(D, axis)
    }
