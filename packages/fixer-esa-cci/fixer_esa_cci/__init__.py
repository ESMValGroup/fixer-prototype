"""A package for making ESA CCI xarray datasets analysis ready."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import yaml


class ESACCIFixer:
    """A fixer for ESA CCI datasets.

    Parameters
    ----------
    fixes:
        A mapping of dataset names to lists of fixes. Each fix is a mapping of
        fix parameters, which must include a `function` key with the fully
        qualified name of the function to apply and the other keys are passed
        as keyword arguments.
    priority:
        The priority of the fixer.

    """

    def __init__(
        self,
        fixes: dict[str, list[dict[str, Any]]],
        priority: int = 10,
    ) -> None:
        self.fixes = fixes
        self.priority = priority

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
        if name is not None and name in self.fixes:
            return self.fixes[name]
        return []

    @classmethod
    def from_defaults(cls) -> Self:
        """Load the default fixes from the YAML file."""
        return cls(
            yaml.safe_load(
                (Path(__file__).parent / "fixes.yaml").read_text(
                    encoding="utf-8",
                ),
            ),
        )


fixer = ESACCIFixer.from_defaults()
