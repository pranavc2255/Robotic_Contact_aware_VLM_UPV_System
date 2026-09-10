"""Future optional semantic contact verifier.

This must remain optional for deployment unless a later phase explicitly makes
it part of the official pipeline.
"""

from __future__ import annotations


def verify_contact_semantics(*_args, **_kwargs):
    raise NotImplementedError("Semantic contact verification is not ported to v2 Phase 1.")

