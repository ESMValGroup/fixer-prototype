"""Protocols for implementing fixer plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import xarray as xr


@runtime_checkable
class FixerPlugin(Protocol):
    """Fixer plugin protocol."""

    priority: int

    def get_fixes(self, name: str | None) -> list[dict[str, Any]]:
        """Get the fixes for a given dataset.

        Parameters
        ----------
        name:
            The name of the dataset to get fixes for.

        Returns
        -------
        :
            A list of fixes, where each fix is a mapping of fix parameters,
            which must include a `function` key with the fully qualified name
            of the function to apply and the other keys are passed as keyword
            arguments to the function.
        """
        ...


@runtime_checkable
class FixFunction(Protocol):
    """Fix function protocol."""

    def __call__(
        self,
        ds: xr.Dataset,
        **kwargs: Any,  # noqa: ANN401
    ) -> xr.Dataset:
        """Call signature of a fix function.

        Parameters
        ----------
        ds:
            The dataset to fix.
        **kwargs:
            Additional keyword arguments that depend on the function.

        Returns
        -------
        :
            The fixed dataset.
        """
        ...
