"""Neutral natural-language identifiers for generated worlds.

The ids produced here are opaque references made from ordinary words.  They are
intended to replace synthetic handles like ``H4ZZ7Y`` or ``doc_594`` without
making the reference semantically helpful.
"""

from __future__ import annotations

import random


DISALLOWED_TOKENS = {
    "alias",
    "anchor",
    "answer",
    "audit",
    "atlas",
    "beacon",
    "block",
    "blocker",
    "bridge",
    "brief",
    "checkpoint",
    "cedar",
    "circuit",
    "code",
    "deadline",
    "decision",
    "delta",
    "doc",
    "document",
    "ember",
    "evidence",
    "falcon",
    "final",
    "handoff",
    "harbor",
    "handle",
    "identity",
    "ion",
    "internal",
    "juniper",
    "keystone",
    "ledger",
    "lumen",
    "market",
    "meridian",
    "memo",
    "nimbus",
    "orbit",
    "orchid",
    "owner",
    "pioneer",
    "policy",
    "portal",
    "pulse",
    "quartz",
    "relay",
    "risk",
    "runway",
    "signal",
    "source",
    "summit",
    "switch",
    "talon",
    "ticket",
    "vertex",
    "vista",
}


ADJECTIVES = [
    "amber", "ancient", "apricot", "arctic", "ash", "autumn", "azure",
    "brass", "bright", "brisk", "bronze", "calm", "cedar", "celadon",
    "cerulean", "clear", "cobalt", "cool", "copper", "coral", "crimson",
    "crystal", "dawn", "deep", "dry", "dusky", "eager", "eastern",
    "ember", "emerald", "even", "fern", "frost", "gentle", "golden",
    "granite", "green", "harbor", "hazel", "hidden", "hollow", "indigo",
    "ivory", "jade", "juniper", "keen", "lacquer", "lilac", "lively",
    "lunar", "maple", "marble", "meadow", "misty", "morning", "moss",
    "navy", "north", "olive", "opal", "orange", "orchid", "pale",
    "pearl", "pine", "plum", "polished", "quiet", "rapid", "red",
    "river", "rose", "royal", "saffron", "sage", "scarlet", "sea",
    "silver", "slate", "smooth", "solar", "south", "spring", "steady",
    "stone", "summer", "sunlit", "teal", "tidal", "umber", "velvet",
    "verdant", "violet", "warm", "western", "white", "wild", "winter",
]


NOUNS = [
    "anchor", "arcade", "archive", "atlas", "basin", "beacon", "brook",
    "canal", "canyon", "cedar", "cellar", "cinder", "circuit", "cliff",
    "copper", "courtyard", "cove", "crest", "delta", "dock", "drift",
    "field", "forge", "garden", "gate", "glade", "harbor", "haven",
    "hearth", "hill", "island", "junction", "kernel", "lagoon", "lantern",
    "ledger", "library", "market", "meadow", "mill", "mirror", "needle",
    "notebook", "oasis", "orbit", "orchard", "passage", "path", "peak",
    "pier", "pine", "plaza", "portal", "quarry", "reef", "ridge", "river",
    "route", "shelf", "signal", "spire", "spring", "station", "stone",
    "summit", "switch", "terrace", "tower", "trail", "valley", "vault",
    "vista", "warehouse", "waypoint", "window", "yard",
]


class NaturalIdBank:
    """Deterministically allocate unique, neutral word slugs."""

    def __init__(self, rng: random.Random, *, used: set[str] | None = None) -> None:
        self.rng = rng
        self.used = set(used or ())

    def fresh(self, *, words: int = 2) -> str:
        """Return a unique lower-case hyphenated phrase."""
        words = max(2, min(3, int(words)))
        for _ in range(10_000):
            if words == 2:
                parts = [self.rng.choice(ADJECTIVES), self.rng.choice(NOUNS)]
            else:
                parts = [
                    self.rng.choice(ADJECTIVES),
                    self.rng.choice(ADJECTIVES),
                    self.rng.choice(NOUNS),
                ]
            if any(part in DISALLOWED_TOKENS for part in parts):
                continue
            candidate = "-".join(parts)
            if candidate not in self.used:
                self.used.add(candidate)
                return candidate
        raise RuntimeError("exhausted natural id candidates")
