"""corpus_trail — answer-first research tasks over a document corpus.

This family is meant to look more like real agent research than the ledger
worlds.  The generator samples a final structured brief, builds a hidden
evidence DAG that proves each field, then renders that DAG into noisy memos,
tickets, policies, and meeting notes.  Some facts are deliberately durable:
they unlock a search/read now and are needed again several hops later for the
final answer.

The model sees only ordinary REPL tools:

- ``search_docs(query, limit=6)`` returns matching ids with short snippets.
- ``read_doc(source_id)`` returns one full verbose document string.
- ``briefing_note`` is a long pre-seeded starting note.

The expected answer is an exact structured value.  No process-level reward is needed for this
family; the existing non-adaptive rubric gives 1.0 only for a fully correct
final answer.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from .base import QueryTemplate, TrainingExample, WorldGenerator
from .natural_ids import NaturalIdBank


CORPUS_TRAIL_TEMPLATES = [
    QueryTemplate(
        name="project_risk_brief",
        description=(
            "Synthesize a current project risk brief from a noisy document "
            "corpus with aliases, stale facts, and durable evidence."
        ),
        min_turns=5,
        max_turns=15,
        answer_type="json",
    ),
]


PROJECT_PREFIXES = [
    "Atlas", "Beacon", "Cedar", "Delta", "Ember", "Falcon", "Harbor",
    "Ion", "Juniper", "Keystone", "Lumen", "Meridian", "Nimbus",
    "Orchid", "Pioneer", "Quartz", "Relay", "Summit", "Talon", "Vertex",
]

PROJECT_SUFFIXES = [
    "Relay", "Bridge", "Ledger", "Portal", "Switch", "Harbor", "Vista",
    "Pulse", "Orbit", "Runway", "Market", "Signal", "Circuit", "Anchor",
]

OWNERS = [
    "Alice Park", "Bob Stone", "Cara Reed", "Evie Patel", "Finn Zhou",
    "Gus Moreno", "Hana Singh", "Ivan Holt", "Joy Rivers", "Kai Tan",
    "Lia Brooks", "Mira Chen", "Nico Vale", "Omar Diaz", "Pia Novak",
]

VENDORS = [
    "Northstar", "Aster Labs", "Blueforge", "Cobalt Lane", "Driftwell",
    "Evergreen Ops", "Fathom Grid", "Glasshouse", "HelioStack", "IrisWorks",
    "Junco Systems", "Kitefield", "LatticePoint", "Monarch Data",
]

BLOCKERS = [
    ("missing SOC2 bridge letter", "blocked"),
    ("regional DPA exception pending", "escalate"),
    ("subprocessor disclosure not countersigned", "blocked"),
    ("production load-test variance lacks owner signoff", "review"),
    ("security questionnaire remediation not accepted", "blocked"),
]

FILLER = [
    "The note also includes routine status language copied from the weekly operations template.",
    "Several paragraphs discuss staffing rotations and do not affect the launch decision.",
    "The author kept the historical summary in place so reviewers could compare earlier assumptions.",
    "A calendar reminder was attached, but the reminder text is not a source of record.",
    "The distribution list includes observers from finance, support, and partner engineering.",
    "Unrelated housekeeping items about dashboard labels were left in the thread for continuity.",
    "The document repeats older language before stating the current source-of-record line.",
    "A formatting appendix was preserved from the previous review packet.",
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _date(day: int) -> str:
    day = max(1, min(28, day))
    return f"2026-06-{day:02d}"


def _code(id_bank: NaturalIdBank) -> str:
    return id_bank.fresh(words=2)


def _related_code(id_bank: NaturalIdBank, code: str) -> str:
    """Create a neutral near-miss code sharing one word with ``code``."""
    left = code.split("-", 1)[0]
    for _ in range(1000):
        right = id_bank.fresh(words=2).split("-")[-1]
        candidate = f"{left}-{right}"
        if candidate not in id_bank.used:
            id_bank.used.add(candidate)
            return candidate
    return id_bank.fresh(words=2)


def _doc_id(used: set[str], id_bank: NaturalIdBank) -> str:
    while True:
        candidate = id_bank.fresh(words=2)
        if candidate not in used:
            used.add(candidate)
            return candidate


def _sample_project(rng: random.Random) -> str:
    return f"{rng.choice(PROJECT_PREFIXES)} {rng.choice(PROJECT_SUFFIXES)}"


def _pad(body: str, rng: random.Random, target_chars: int) -> str:
    parts = [body]
    while len("\n".join(parts)) < target_chars:
        parts.append(rng.choice(FILLER))
    return "\n".join(parts)


def _render_doc(doc: dict[str, Any]) -> str:
    keywords = ", ".join(doc.get("keywords", []))
    return (
        f"Document {doc['id']}\n"
        f"Title: {doc['title']}\n"
        f"Date: {doc['date']}\n"
        f"Keywords: {keywords}\n\n"
        f"{doc['body']}"
    )


def _tokenize(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) >= 2 and t not in {"the", "and", "for", "with", "from"}
    ]


class CorpusTrailWorld(WorldGenerator):
    world_type = "corpus_trail"

    def get_query_templates(self) -> list[QueryTemplate]:
        return list(CORPUS_TRAIL_TEMPLATES)

    def get_system_prompt(self) -> str:
        return (
            "corpus_trail world: search and read a noisy document corpus, then "
            "submit one exact structured brief."
        )

    def generate_state(
        self,
        depth: int,
        breadth: int,
        rng: random.Random,
        *,
        difficulty: int | None = None,
    ) -> dict:
        difficulty = max(0, min(4, difficulty if difficulty is not None else 2))
        params = {
            0: {
                "distractors": 1,
                "cap": 2200,
                "doc_chars": 280,
                "lag": 2,
                "briefing_noise": 0,
                "snippet_chars": 220,
                "near_miss_rate": 0.0,
                "risk_public": False,
                "stale_docs": False,
                "cross_refs": False,
            },
            1: {
                "distractors": 1,
                "cap": 1500,
                "doc_chars": 380,
                "lag": 2,
                "briefing_noise": 0,
                "snippet_chars": 220,
                "near_miss_rate": 0.05,
                "risk_public": False,
                "stale_docs": True,
                "cross_refs": False,
            },
            2: {
                "distractors": 3,
                "cap": 1650,
                "doc_chars": 540,
                "lag": 3,
                "briefing_noise": 2,
                "snippet_chars": 180,
                "near_miss_rate": 0.10,
                "risk_public": False,
                "stale_docs": True,
                "cross_refs": False,
            },
            3: {
                "distractors": 10,
                "cap": 1750,
                "doc_chars": 620,
                "lag": 4,
                "briefing_noise": 7,
                "snippet_chars": 150,
                "near_miss_rate": 0.16,
                "risk_public": False,
                "stale_docs": True,
                "cross_refs": False,
            },
            4: {
                "distractors": 22,
                "cap": 1850,
                "doc_chars": 760,
                "lag": 5,
                "briefing_noise": 14,
                "snippet_chars": 140,
                "near_miss_rate": 0.24,
                "risk_public": False,
                "stale_docs": True,
                "cross_refs": False,
            },
        }[difficulty]

        id_bank = NaturalIdBank(rng)
        used_doc_ids: set[str] = set()
        project = _sample_project(rng)
        internal_code = _code(id_bank)
        vendor_alias = rng.choice(VENDORS)
        owner = rng.choice(OWNERS)
        old_owner = rng.choice([o for o in OWNERS if o != owner])
        blocker_pool = (
            [item for item in BLOCKERS if item[1] == "blocked"]
            if difficulty <= 1
            else BLOCKERS
        )
        project_tokens = set(_tokenize(project))
        blocker_candidates = [
            item for item in blocker_pool
            if not (project_tokens & set(_tokenize(item[0])))
        ]
        blocker, decision = rng.choice(blocker_candidates or blocker_pool)
        old_decision = rng.choice([d for d in ["ready", "blocked", "review", "escalate"] if d != decision])
        deadline_day = rng.randint(12, 25)
        old_deadline_day = max(2, deadline_day - rng.randint(3, 8))
        deadline = _date(deadline_day)
        old_deadline = _date(old_deadline_day)
        policy_id = id_bank.fresh(words=2)
        ticket_id = id_bank.fresh(words=2)
        structured_sources = difficulty <= 2
        neutral_surface = True

        def source_body(fields: list[tuple[str, str]], narrative: str) -> str:
            if not structured_sources:
                return narrative
            header = "\n".join(f"{name}: {value}" for name, value in fields)
            return f"{header}\n{narrative}"

        docs: dict[str, dict[str, Any]] = {}

        def add_doc(prefix: str, title: str, date: str, body: str, keywords: list[str], *, target: int | None = None) -> str:
            doc_id = _doc_id(used_doc_ids, id_bank)
            docs[doc_id] = {
                "id": doc_id,
                "title": title,
                "date": date,
                "body": _pad(body, rng, target or params["doc_chars"]),
                "keywords": keywords,
            }
            return doc_id

        def title(role: str) -> str:
            if not neutral_surface:
                return {
                    "identity": f"Alias registry for {project}",
                    "old_owner": f"Older launch owner note for {internal_code}",
                    "owner": f"Current ownership and date for {internal_code}",
                    "old_ticket": f"Historical ticket status for {vendor_alias}",
                    "blocker": f"Evidence ticket {ticket_id} for {internal_code}",
                    "policy": f"{policy_id} launch risk decision policy",
                }[role]
            return {
                "identity": f"Reference note for {project}",
                "old_owner": f"Earlier planning note {internal_code}",
                "owner": f"June routing note {internal_code}",
                "old_ticket": f"Prior vendor status {ticket_id}",
                "blocker": f"Vendor issue packet {ticket_id}",
                "policy": f"Decision standard {policy_id}",
            }[role]

        def owner_narrative() -> str:
            if neutral_surface:
                return (
                    f"Current launch handoff for {internal_code}: owner is {owner}. "
                    f"The active deadline is {deadline}. This supersedes older "
                    f"ownership notes for this launch stream and applies to the "
                    f"vendor alias {vendor_alias}."
                )
            return (
                f"Current launch handoff for {internal_code}: owner is {owner}. "
                f"The active deadline is {deadline}. This supersedes older "
                f"ownership notes for {project} and applies to the vendor alias "
                f"{vendor_alias}."
            )

        identity_id = add_doc(
            "memo",
            title("identity"),
            _date(2),
            source_body(
                [
                    ("Project", project),
                    ("Internal code", internal_code),
                    ("Vendor alias", vendor_alias),
                ],
                f"Source-of-record alias mapping: public project {project} is "
                f"tracked internally as {internal_code}. Vendor-channel notes "
                f"use the alias {vendor_alias}. Preserve this mapping because "
                f"later launch, ticket, and risk documents may use only one of "
                f"the three names.",
            ),
            [project, internal_code, vendor_alias, "alias", "registry"],
        )

        stale_doc_ids: list[str] = []
        if params["stale_docs"]:
            stale_owner_id = add_doc(
                "email",
                title("old_owner"),
                _date(5),
                (
                    f"Earlier planning note for {internal_code}: {old_owner} was "
                    f"listed as launch owner and the target date was {old_deadline}. "
                    f"This note predates the current handoff and should be treated "
                    f"as historical unless a later source repeats it."
                ),
                [internal_code, old_owner, old_deadline, "historical"],
                target=params["doc_chars"] - 80,
            )
            stale_doc_ids.append(stale_owner_id)

        owner_id = add_doc(
            "handoff",
            title("owner"),
            _date(11),
            source_body(
                [
                    ("Internal code", internal_code),
                    ("Owner", owner),
                    ("Deadline", deadline),
                    ("Vendor alias", vendor_alias),
                ],
                owner_narrative(),
            ),
            (
                [internal_code, owner, deadline, vendor_alias, "handoff"]
                if neutral_surface
                else [internal_code, project, owner, deadline, vendor_alias, "handoff"]
            ),
        )

        if params["stale_docs"]:
            stale_ticket_id = add_doc(
                "ticket",
                title("old_ticket"),
                _date(12),
                (
                    f"Historical status update for {ticket_id}: launch looked "
                    f"{old_decision} after an initial evidence sweep for "
                    f"{vendor_alias}. This update did not apply {policy_id} and was "
                    f"recorded before the final risk review."
                ),
                [ticket_id, vendor_alias, old_decision, "historical"],
                target=params["doc_chars"] - 60,
            )
            stale_doc_ids.append(stale_ticket_id)

        blocker_id = add_doc(
            "ticket",
            title("blocker"),
            _date(14),
            source_body(
                [
                    ("Ticket", ticket_id),
                    ("Internal code", internal_code),
                    ("Vendor alias", vendor_alias),
                    ("Blocker", blocker),
                    ("Policy", policy_id),
                ],
                f"Open evidence ticket {ticket_id} for {internal_code}: the "
                f"blocking issue is {blocker}. The ticket is filed under vendor "
                f"alias {vendor_alias} and says classification must follow "
                f"{policy_id} rather than the historical ticket status.",
            ),
            [ticket_id, internal_code, vendor_alias, blocker, policy_id],
        )

        policy_id_doc = add_doc(
            "policy",
            title("policy"),
            _date(16),
            source_body(
                [
                    ("Policy", policy_id),
                    ("Blocker", blocker),
                    ("Decision", decision),
                ],
                f"Policy {policy_id}: when the active blocker is '{blocker}', "
                f"the launch decision is '{decision}'. If an old ticket status "
                f"conflicts with a later risk memo, the later risk memo is the "
                f"decision source. Keep the internal code and vendor alias "
                f"attached to the decision so similarly named projects are not "
                f"merged.",
            ),
            [policy_id, blocker, decision, "risk", "decision"],
        )

        if params["risk_public"]:
            risk_title = f"Final risk memo for {project} / {vendor_alias} ticket {ticket_id}"
        elif neutral_surface:
            risk_title = f"Board disposition packet {vendor_alias} {ticket_id}"
        else:
            risk_title = f"Final risk memo for {vendor_alias} ticket {ticket_id}"
        cross_ref_line = (
            "Evidence cross-reference: use "
            f"{identity_id} for identity, {owner_id} for ownership/date, "
            f"{blocker_id} for the active blocker, {policy_id_doc} for policy, "
            "and this risk memo as the final decision source. "
            if params["cross_refs"]
            else ""
        )
        risk_body = (
            source_body(
                (
                    [
                        (
                            "Evidence order",
                            f"{identity_id}, {owner_id}, {blocker_id}, "
                            f"{policy_id_doc}, this risk memo",
                        ),
                        ("Project", project),
                        ("Internal code", internal_code),
                        ("Decision", decision),
                        ("Source type", "final risk memo"),
                        ("Vendor alias", vendor_alias),
                        ("Ticket", ticket_id),
                        ("Policy", policy_id),
                        ("Blocker", blocker),
                        ("Identity evidence", identity_id),
                        ("Owner/deadline evidence", owner_id),
                        ("Blocker evidence", blocker_id),
                        ("Policy evidence", policy_id_doc),
                        ("Final memo evidence", "this document"),
                    ]
                    if params["cross_refs"]
                    else [
                    ("Project", project),
                    ("Internal code", internal_code),
                    ("Decision", decision),
                    ("Source type", "final risk memo"),
                    ("Vendor alias", vendor_alias),
                    ("Ticket", ticket_id),
                    ("Policy", policy_id),
                    ("Blocker", blocker),
                    ]
                ),
                f"Final risk board memo for public project {project}, internal "
                f"code {internal_code}, vendor alias {vendor_alias}, and ticket "
                f"{ticket_id}: applying {policy_id}, the current launch decision "
                f"is {decision}; the active blocker issue is {blocker}. "
                f"This memo is the final decision source for the "
                f"evidence list. {cross_ref_line}It intentionally does not restate the launch "
                f"owner or deadline; use the current handoff source for those "
                f"fields.",
            )
            if params["risk_public"]
            else (
                f"Final risk board memo for vendor alias {vendor_alias} and "
                f"ticket {ticket_id}: applying {policy_id}, the current "
                f"launch decision is {decision}. This memo is the final "
                f"decision source for the evidence list. It intentionally "
                f"does not restate the launch owner or deadline; use the "
                f"current handoff source for those fields."
            )
        )
        risk_keywords = (
            [project, internal_code, vendor_alias, policy_id, ticket_id, decision, "risk", "final"]
            if params["risk_public"]
            else [vendor_alias, policy_id, ticket_id, decision, "risk", "final"]
        )
        decision_id = add_doc(
            "risk",
            risk_title,
            _date(20),
            risk_body,
            risk_keywords,
        )

        # Distractors: other projects, stale updates, and near collisions.
        for _ in range(params["distractors"]):
            other_project = _sample_project(rng)
            other_code = _code(id_bank)
            other_vendor = rng.choice(VENDORS)
            other_owner = rng.choice(OWNERS)
            other_blocker, other_decision = rng.choice(BLOCKERS)
            kind = rng.choice(["memo", "ticket", "risk", "handoff", "policy"])
            if rng.random() < params["near_miss_rate"]:
                # Near-miss docs share one target token but point elsewhere.
                other_vendor = vendor_alias if rng.random() < 0.5 else other_vendor
                other_code = _related_code(id_bank, internal_code)
            distractor_title = rng.choice([
                f"{kind.title()} note for {other_project}",
                f"{other_code} {kind} update",
                f"{other_vendor} historical packet",
            ])
            if neutral_surface:
                distractor_title = rng.choice([
                    f"Reference note {other_code}",
                    f"Vendor issue packet {rng.choice([ticket_id, id_bank.fresh(words=2)])}",
                    f"Board packet {other_vendor}",
                    f"Decision standard {id_bank.fresh(words=2)}",
                ])
            if neutral_surface:
                body = (
                    f"This {kind} concerns {other_project}. It may mention "
                    f"{other_vendor}, {other_code}, owner {other_owner}, blocker "
                    f"{other_blocker}, and decision {other_decision}. Treat it "
                    "as unrelated unless a source-of-record alias mapping connects "
                    "the identifiers."
                )
            else:
                body = (
                    f"This {kind} concerns {other_project}. It may mention "
                    f"{other_vendor}, {other_code}, owner {other_owner}, "
                    f"blocker {other_blocker}, and decision {other_decision}. "
                    "Treat it as unrelated unless a source-of-record alias "
                    "mapping connects the identifiers."
                )
            add_doc(
                kind,
                distractor_title,
                _date(rng.randint(1, 24)),
                body,
                [other_project, other_code, other_vendor, other_owner, other_decision],
                target=max(420, params["doc_chars"] - rng.randint(80, 220)),
            )

        # A long initial document that gives useful starting clues but is not a
        # citable source in the final answer.
        briefing_lines = [
            "Research intake briefing. Use this as a starting note, not as a final evidence source.",
            (
                f"The user-facing project name is {project}. The alias registry should "
                f"confirm that it maps to {internal_code} and vendor alias {vendor_alias}."
                if not neutral_surface
                else (
                    f"The user-facing project name is {project}. Start by finding the "
                    "source-of-record mapping; later documents may use only the internal "
                    "code or vendor alias, and those identifiers are needed again near "
                    "the end."
                )
            ),
            (
                f"Compliance classifications for this batch reference {policy_id}; "
                "the policy may be needed again after you inspect the ticket and risk memo."
                if not neutral_surface
                else (
                    "The active ticket names the policy standard. Keep the policy id "
                    "and blocker together because the final board packet may only repeat "
                    "part of that chain."
                )
            ),
            "The final evidence list has five source documents: alias registry, current handoff, active ticket, policy, and final risk memo.",
            "The final risk memo is often filed under the vendor alias, ticket id, or policy id rather than the public project name.",
            "Several older tickets can look authoritative but were copied from earlier planning packets.",
        ]
        for _ in range(params["briefing_noise"]):
            other_project = _sample_project(rng)
            briefing_lines.append(
                f"Noise note: {other_project} has unrelated code {_code(id_bank)} "
                f"and owner {rng.choice(OWNERS)}."
            )
        briefing_doc = "\n".join(briefing_lines)

        answer = {
            "project": project,
            "internal_code": internal_code,
            "owner": owner,
            "blocker": blocker,
            "deadline": deadline,
            "decision": decision,
            "evidence": [
                identity_id,
                owner_id,
                blocker_id,
                policy_id_doc,
                decision_id,
            ],
        }

        gold_doc_ids = [identity_id, owner_id, blocker_id, policy_id_doc, decision_id]
        raw_gold_chars = sum(len(_render_doc(docs[d])) for d in gold_doc_ids) + len(briefing_doc)
        compact = (
            f"{project}={internal_code}={vendor_alias}; owner {owner}; "
            f"deadline {deadline}; blocker {blocker}; {policy_id}->{decision}; "
            f"src {','.join(gold_doc_ids)}"
        )
        compact_chars = len(compact)

        facts = [
            {
                "fact_id": "alias_map",
                "source": identity_id,
                "roles": ["route_unlock", "final_answer", "evidence"],
                "first_used_hop": 1,
                "needed_again_hop": 5 + params["lag"],
                "value": f"{project} = {internal_code} = {vendor_alias}",
            },
            {
                "fact_id": "policy_rule",
                "source": policy_id_doc,
                "roles": ["interpretation_rule", "final_decision"],
                "first_used_hop": 4,
                "needed_again_hop": 7 + params["lag"],
                "value": f"{policy_id}: {blocker} -> {decision}",
            },
            {
                "fact_id": "owner_deadline",
                "source": owner_id,
                "roles": ["final_answer", "stale_override"],
                "first_used_hop": 3,
                "needed_again_hop": 8,
                "value": f"{owner}, {deadline}",
            },
        ]

        return {
            "docs": docs,
            "briefing_note": briefing_doc,
            "document_ids": sorted(docs),
            "max_context_chars": params["cap"],
            "search_snippet_chars": params["snippet_chars"],
            "_answer": answer,
            "_difficulty": difficulty,
            "_gold_doc_ids": gold_doc_ids,
            "_durable_facts": facts,
            "_raw_gold_chars": raw_gold_chars,
            "_compact_gold_chars": compact_chars,
            "_append_only_margin": round(raw_gold_chars / params["cap"], 2),
            "_stale_doc_ids": stale_doc_ids,
            "_route_terms": {
                "internal_code": internal_code,
                "vendor_alias": vendor_alias,
                "ticket_id": ticket_id,
                "policy_id": policy_id,
            },
        }

    def generate_query(
        self,
        template: QueryTemplate,
        state: dict,
        target_turns: int,
        rng: random.Random,
    ) -> tuple[str, Any, int, int]:
        rich = self.generate_query_rich(template, state, target_turns, rng)
        return (
            rich["query_text"],
            rich["expected_answer"],
            rich["actual_depth"],
            rich["actual_breadth"],
        )

    def generate_query_rich(
        self,
        template: QueryTemplate,
        state: dict,
        target_turns: int,
        rng: random.Random,
    ) -> dict:
        answer = state["_answer"]
        project = answer["project"]
        difficulty = int(state.get("_difficulty", 2))
        if difficulty == 0:
            expected_answer: Any = list(answer["evidence"])
            query_template = "evidence_trail"
            query = (
                f"Find the ordered evidence trail for project '{project}'. "
                "The Python variable briefing_note is already seeded with a long "
                "intake note. Use search_docs(query, limit=6) and read_doc(source_id) "
                "to inspect the corpus. Search results are snippets only; read the "
                "source documents you rely on. Return exactly one list of "
                "five document ids, ordered as identity, owner/deadline, blocker, "
                "policy, and final risk memo. The policy is not a substitute for "
                "the final risk memo evidence source. Do not cite briefing_note as "
                "evidence. Submit via submit_answer(value)."
            )
        elif difficulty == 1:
            expected_answer = {
                key: answer[key]
                for key in (
                    "project",
                    "internal_code",
                    "evidence",
                )
            }
            query_template = "compact_risk_brief"
            query = (
                f"Prepare the compact current launch risk brief for project '{project}'. "
                "The Python variable briefing_note is already seeded with a long "
                "intake note. Use search_docs(query, limit=6) and read_doc(source_id) "
                "to inspect the corpus. Search results are snippets only; read the "
                "source documents you rely on. Return exactly one object with "
                "keys project, internal_code, evidence. "
                "evidence must be the ordered list of five document ids supporting "
                "identity, owner/deadline, blocker, policy, and the final risk memo. "
                "The policy is not a substitute for the final risk memo evidence "
                "source. Do not cite briefing_note as evidence. Submit via "
                "submit_answer(value)."
            )
        elif difficulty == 2:
            expected_answer = {
                key: answer[key]
                for key in (
                    "project",
                    "internal_code",
                    "owner",
                    "deadline",
                    "decision",
                    "evidence",
                )
            }
            query_template = "handoff_risk_brief"
            query = (
                f"Prepare the current handoff risk brief for project '{project}'. "
                "The Python variable briefing_note is already seeded with a long "
                "intake note. Use search_docs(query, limit=6) and read_doc(source_id) "
                "to inspect the corpus. Search results are snippets only; read the "
                "source documents you rely on. Return exactly one object with "
                "keys project, internal_code, owner, deadline, decision, evidence. "
                "evidence must be the ordered list of five document ids supporting "
                "identity, owner/deadline, blocker ticket, policy, and the final "
                "risk memo. The policy is not a substitute for the final risk memo "
                "evidence source. Do not cite briefing_note as evidence. Submit via "
                "submit_answer(value)."
            )
        else:
            expected_answer = answer
            query_template = "project_risk_brief"
            query = (
                f"Prepare the current launch risk brief for project '{project}'. "
                "The Python variable briefing_note is already seeded with a long "
                "intake note. Use search_docs(query, limit=6) and read_doc(source_id) "
                "to inspect the corpus. Search results are snippets only; read the "
                "source documents you rely on. Return exactly one object with "
                "keys project, internal_code, owner, blocker, "
                "deadline, decision, evidence. evidence must be the ordered list of "
                "five document ids supporting identity, owner/deadline, blocker, "
                "policy, and the final risk memo. The policy is not a substitute "
                "for the final risk memo evidence source. Do not cite briefing_note "
                "as evidence. Submit via submit_answer(value)."
            )
        return {
            "query_text": query,
            "expected_answer": json.dumps(expected_answer, separators=(",", ":")),
            "actual_depth": len(state.get("_gold_doc_ids", [])),
            "actual_breadth": len(state.get("docs", {})),
            "query_template": query_template,
            "target_entities": list(state.get("_gold_doc_ids", [])),
            "parameter_refs": [],
            "rules": [],
        }


def make_example(seed: int, difficulty: int) -> TrainingExample:
    rng = random.Random(seed)
    gen = CorpusTrailWorld()
    template = gen.get_query_templates()[0]
    state = gen.generate_state(depth=5, breadth=20, rng=rng, difficulty=difficulty)
    rich = gen.generate_query_rich(template, state, target_turns=9, rng=rng)
    return TrainingExample(
        example_id=rng.randint(0, 2**31 - 1),
        world_type=gen.world_type,
        system_prompt=gen.get_system_prompt(),
        user_query=rich["query_text"],
        # Keep underscore-prefixed provenance metadata in the dataset row for
        # audits. ContextToolsEnv strips these keys before writing the sandbox
        # context, so the model only receives rendered docs and public helpers.
        state=dict(state),
        optimal_turns=8 + difficulty,
        expected_answer=rich["expected_answer"],
        answer_type=template.answer_type,
        difficulty=difficulty,
        depth=rich["actual_depth"],
        breadth=rich["actual_breadth"],
        query_template=str(rich.get("query_template") or template.name),
        target_entities=list(rich.get("target_entities", [])),
        parameter_refs=[],
        rules=[],
    )
