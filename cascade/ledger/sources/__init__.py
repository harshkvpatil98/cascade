"""Source loaders. Each normalises one upstream into ``RawQuestion``.

A loader's only job is normalisation: parse, coerce timestamps to aware UTC,
and refuse anything it cannot read. Screening happens later, in ``rules.py``,
so that the inclusion rules are applied identically to every source rather
than reimplemented four times with four sets of edge cases.

A loader never invents a field. A missing or unparseable timestamp drops the
question rather than defaulting -- the same rule the corpus applies to
``published_at`` at M2, for the same reason.
"""

from __future__ import annotations

__all__: list[str] = []
