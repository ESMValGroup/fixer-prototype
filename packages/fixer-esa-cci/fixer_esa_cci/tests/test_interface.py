from fixer import FIXERS
from fixer.protocol import FixerPlugin

from fixer_esa_cci import fixer


def test_fixer() -> None:
    assert isinstance(fixer, FixerPlugin)
    assert fixer in FIXERS
