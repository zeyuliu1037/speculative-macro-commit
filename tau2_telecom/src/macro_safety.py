"""Semantic macro-family guards for TAU2 telecom macro skipping.

These helpers are intentionally independent of ``tau2`` imports so they can
be tested in the base repo environment.  Certificates still audit replay/tool
execution; this module only decides whether a macro family is semantically
eligible to fire for the task intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MACRO_FAMILIES = {
    "diagnostic_lookup",
    "network_diagnosis",
    "billing_action",
    "line_state_mutation",
    "account_contract_mutation",
}

_BILLING_ACTIONS = {
    "make_payment",
    "send_payment_request",
    "check_payment_request",
    "refuel_data",
}
_LINE_STATE_ACTIONS = {
    "resume_line",
    "suspend_line",
    "pause_line",
    "activate_line",
    "deactivate_line",
}
_ACCOUNT_CONTRACT_MUTATION_HINTS = (
    "contract",
    "account",
    "plan",
)
_NETWORK_ACTION_HINTS = (
    "apn",
    "airplane",
    "app_permission",
    "data",
    "mms",
    "network",
    "permission",
    "reboot",
    "roaming",
    "sim",
    "speed",
    "vpn",
    "wifi",
)

_BILLING_INTENT_HINTS = (
    "bill",
    "billing",
    "invoice",
    "overdue",
    "pay",
    "payment",
    "refuel",
    "usage_exceeded",
)
_LINE_STATE_INTENT_HINTS = (
    "line",
    "resume",
    "suspend",
    "suspension",
    "contract_end_suspension",
    "overdue_bill_suspension",
)
_ACCOUNT_CONTRACT_INTENT_HINTS = (
    "account",
    "contract",
    "contract_end",
    "plan",
)


@dataclass(frozen=True)
class SemanticGuardResult:
    allowed: bool
    reason: str
    macro_family: str
    safety_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "macro_family": self.macro_family,
            "safety_class": self.safety_class,
        }


def normalize_task_intent_text(task: Any) -> str:
    """Extract stable task-intent text without depending on tau2 schemas."""
    parts: list[str] = []
    for attr in ("id", "instruction", "description", "goal", "user_goal"):
        value = getattr(task, attr, None)
        if value:
            parts.append(str(value))
    if not parts and isinstance(task, dict):
        for key in ("id", "instruction", "description", "goal", "user_goal"):
            value = task.get(key)
            if value:
                parts.append(str(value))
    return " ".join(parts).lower()


def classify_macro_family(actions: list[str]) -> str:
    action_set = {str(action) for action in actions}
    lowered = " ".join(action_set).lower()
    if action_set & _LINE_STATE_ACTIONS:
        return "line_state_mutation"
    if any(hint in lowered for hint in _ACCOUNT_CONTRACT_MUTATION_HINTS):
        mutating_contract = any(
            action.startswith(("cancel_", "change_", "create_", "renew_", "set_", "update_"))
            and any(hint in action for hint in _ACCOUNT_CONTRACT_MUTATION_HINTS)
            for action in action_set
        )
        if mutating_contract:
            return "account_contract_mutation"
    if action_set & _BILLING_ACTIONS:
        return "billing_action"
    if any(hint in lowered for hint in _NETWORK_ACTION_HINTS):
        return "network_diagnosis"
    return "diagnostic_lookup"


def safety_class_for_family(macro_family: str) -> str:
    if macro_family == "diagnostic_lookup":
        return "read_only"
    if macro_family == "network_diagnosis":
        return "guarded_network_action"
    if macro_family in {"billing_action", "line_state_mutation", "account_contract_mutation"}:
        return "semantic_guarded_mutation"
    return "semantic_guarded_unknown"


def annotate_pattern_safety(pattern: dict[str, Any],
                            actions: list[str]) -> dict[str, Any]:
    item = dict(pattern)
    family = str(item.get("macro_family") or classify_macro_family(actions))
    if family not in MACRO_FAMILIES:
        family = classify_macro_family(actions)
    item["macro_family"] = family
    item["safety_class"] = item.get("safety_class") or safety_class_for_family(family)
    return item


def semantic_family_guard(pattern: dict[str, Any],
                          task_intent_text: str) -> SemanticGuardResult:
    actions = [str(action) for action in (pattern.get("_actions") or pattern.get("action_types") or [])]
    family = str(pattern.get("macro_family") or classify_macro_family(actions))
    safety_class = str(pattern.get("safety_class") or safety_class_for_family(family))
    text = (task_intent_text or "").lower()

    def has_any(hints: tuple[str, ...]) -> bool:
        return any(hint in text for hint in hints)

    action_set = set(actions)
    if "make_payment" in action_set and not has_any(_BILLING_INTENT_HINTS):
        return SemanticGuardResult(False, "make_payment_without_billing_intent", family, safety_class)
    if (action_set & (_LINE_STATE_ACTIONS | {"resume_line", "suspend_line"})
            and not has_any(_LINE_STATE_INTENT_HINTS)):
        return SemanticGuardResult(False, "line_state_without_line_intent", family, safety_class)
    if family == "account_contract_mutation" and not has_any(_ACCOUNT_CONTRACT_INTENT_HINTS):
        return SemanticGuardResult(False, "account_contract_without_explicit_intent", family, safety_class)
    if family == "billing_action" and (action_set & {"send_payment_request", "refuel_data"}):
        if not has_any(_BILLING_INTENT_HINTS):
            return SemanticGuardResult(False, "billing_action_without_billing_intent", family, safety_class)
    return SemanticGuardResult(True, "allowed", family, safety_class)
