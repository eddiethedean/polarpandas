"""
Exception conversion utilities for PolarPandas.

This module provides utilities to convert Polars exceptions to pandas-compatible
exceptions for better API compatibility, including helpful error messages with
typo detection.
"""

from typing import List, Type, TypeVar

T = TypeVar("T", bound=Exception)


def _calculate_similarity(s1: str, s2: str) -> float:
    """Calculate simple string similarity (0.0 to 1.0).

    Uses Levenshtein-like distance normalized by string length.
    """
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    # Simple character overlap
    s1_chars = set(s1.lower())
    s2_chars = set(s2.lower())
    if not s1_chars or not s2_chars:
        return 0.0

    intersection = len(s1_chars & s2_chars)
    union = len(s1_chars | s2_chars)
    if union == 0:
        return 0.0

    return intersection / union


def _find_similar_names(
    name: str, candidates: List[str], max_suggestions: int = 3
) -> List[str]:
    """Find similar names in candidates list using simple similarity.

    Parameters
    ----------
    name : str
        The name to find matches for
    candidates : List[str]
        List of candidate names
    max_suggestions : int, default 3
        Maximum number of suggestions to return

    Returns
    -------
    List[str]
        List of similar names, sorted by similarity (most similar first)
    """
    similarities = [
        (candidate, _calculate_similarity(name, candidate)) for candidate in candidates
    ]
    # Sort by similarity (descending), then alphabetically
    similarities.sort(key=lambda x: (-x[1], x[0]))
    # Filter out exact matches and very low similarity
    similar = [
        candidate for candidate, sim in similarities if sim > 0.3 and candidate != name
    ]
    return similar[:max_suggestions]


def create_keyerror_with_suggestions(
    key: str, available_keys: List[str], context: str = "column"
) -> KeyError:
    """
    Create a KeyError with helpful suggestions for typos.

    Parameters
    ----------
    key : str
        The key that was not found
    available_keys : List[str]
        List of available keys (e.g., column names)
    context : str, default "column"
        Context for the error message (e.g., "column", "index")

    Returns
    -------
    KeyError
        KeyError with helpful message including suggestions

    Examples
    --------
    >>> raise create_keyerror_with_suggestions("name", ["name_", "value", "data"])
    KeyError: "'name' not found. Did you mean 'name_'?"
    """
    message = f"'{key}'"
    suggestions = _find_similar_names(key, available_keys)

    if suggestions:
        if len(suggestions) == 1:
            message += f" not found. Did you mean '{suggestions[0]}'?"
        else:
            suggestions_str = "', '".join(suggestions)
            message += f" not found. Did you mean one of: '{suggestions_str}'?"
    else:
        message += " not found."

    # Match pandas error format for KeyError on columns
    if available_keys:
        message += f" Available {context}s: {', '.join(sorted(available_keys)[:10])}"
        if len(available_keys) > 10:
            message += f" (and {len(available_keys) - 10} more)"

    return KeyError(message)


def convert_polars_exception(
    e: Exception,
    target_type: Type[T] = KeyError,  # type: ignore[assignment]
) -> T:
    """
    Convert Polars exceptions to pandas-compatible ones.

    Parameters
    ----------
    e : Exception
        The original exception from Polars
    target_type : type, default KeyError
        The target exception type to convert to

    Returns
    -------
    Exception
        Converted exception of target_type, or original exception if no conversion needed

    Examples
    --------
    >>> try:
    ...     df["nonexistent"]
    ... except Exception as e:
    ...     raise convert_polars_exception(e, KeyError)
    """
    error_str = str(e).lower()
    exception_type_str = str(type(e))

    # Check if it's a column/key not found error
    if "not found" in error_str or "ColumnNotFoundError" in exception_type_str:
        return target_type(str(e))

    # Re-raise original exception if no conversion needed
    return e  # type: ignore[return-value]


def convert_to_keyerror(e: Exception) -> KeyError:
    """
    Convert exception to KeyError (common case for column/key errors).

    Parameters
    ----------
    e : Exception
        The original exception

    Returns
    -------
    KeyError
        Converted KeyError exception

    Examples
    --------
    >>> try:
    ...     df["missing_col"]
    ... except Exception as e:
    ...     raise convert_to_keyerror(e)
    """
    return convert_polars_exception(e, KeyError)
