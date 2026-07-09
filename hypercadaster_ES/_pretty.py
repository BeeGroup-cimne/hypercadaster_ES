"""Pretty-reprinting containers.

Large module-level data structures (lookup dicts, tag taxonomies, field
layouts) are otherwise rendered by ``repr()`` — and therefore by the pdoc API
docs — as a single unreadable line. Wrapping them in these thin ``dict`` /
``list`` / ``tuple`` subclasses only changes ``__repr__`` (to a ``pprint``-style
indented form); every other behaviour is identical, so they are drop-in
replacements for the built-in containers.
"""
import pprint

__all__ = ["PrettyDict", "PrettyList", "PrettyTuple"]


def _pformat(obj) -> str:
    # sort_dicts=False keeps the authored key order, which is meaningful for
    # the field-layout / taxonomy constants in this package.
    return pprint.pformat(obj, width=100, sort_dicts=False)


class PrettyDict(dict):
    """A ``dict`` whose ``repr`` is pretty-printed (indented, one entry group
    per line) so it reads well in the generated API docs."""
    def __repr__(self) -> str:
        return _pformat(dict(self))


class PrettyList(list):
    """A ``list`` whose ``repr`` is pretty-printed for readable API docs."""
    def __repr__(self) -> str:
        return _pformat(list(self))


class PrettyTuple(tuple):
    """A ``tuple`` whose ``repr`` is pretty-printed for readable API docs."""
    def __repr__(self) -> str:
        return _pformat(tuple(self))
