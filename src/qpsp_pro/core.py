"""Import alias for the canonical implementation (uses the current API).

The former compact simulation API is no longer implemented here. See the
README migration notes before upgrading calls written for version 0.1.0.
"""

from .qpsp_pro import QpspProResult, qpsp_pro

__all__ = ["qpsp_pro", "QpspProResult"]
