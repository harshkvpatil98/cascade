"""Evidence corpus pipeline (M2, spec §3.2).

``fetch -> normalize -> date-validate -> dedupe -> chunk -> embed -> index``
over >= 1.30M pre-cutoff chunks. The date-validate stage drops any document
whose ``published_at`` is null, future or timezone-naive: an uncertain date is
a leakage vector, and inferring one is how the study gets silently invalidated.
"""

from __future__ import annotations

__all__: list[str] = []
