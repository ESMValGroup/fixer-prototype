from fixer import FIXERS
from fixer.protocol import FixerPlugin

from fixer_cmip7 import fixer


def test_fixer() -> None:
    assert isinstance(fixer, FixerPlugin)
    assert fixer in FIXERS
