"""Classify AppWorld API actions as read-only or mutating.

Read-only: safe to execute during speculative lookahead (no env mutation).
Mutating: must defer to Phase F (only execute once, after verification).
"""

# Actions that modify environment state
MUTATING_KEYWORDS = frozenset([
    "login", "logout", "signup", "delete", "create", "update", "send",
    "add_", "remove", "follow", "unfollow", "reset", "complete",
    "clear", "apply", "cancel", "approve", "assign", "accept",
    "reject", "forward", "reply", "label", "compress", "decompress",
    "copy", "move", "rename", "debit", "credit", "like", "dislike",
    "mark", "place", "record", "upload", "pay", "charge", "transfer",
    "withdraw", "deposit", "archive", "star", "pin", "mute", "block",
    "rate", "close", "submit",
])

# Explicit read-only actions (override keyword matching)
READ_ONLY_ACTIONS = frozenset([
    "show_active_task", "show_account_passwords", "show_profile",
    "show_addresses", "show_payment_cards",
    "show_app_descriptions", "show_api_descriptions", "show_api_doc",
    "search_api_docs",
])


def is_read_only(action_name: str) -> bool:
    """Check if an action is read-only (safe for speculative execution).

    Args:
        action_name: Either a plain tool name ("show_profile") or
                     app-qualified name ("spotify.show_song").
    """
    # Extract the api part from "app.api_name"
    api = action_name.split(".")[-1] if "." in action_name else action_name

    # Check explicit read-only list
    if api in READ_ONLY_ACTIONS:
        return True

    # Check for "show_" or "search_" prefix (almost always read-only)
    if api.startswith("show_") or api.startswith("search_") or api.startswith("get_"):
        return True

    # Check for mutating keywords
    for kw in MUTATING_KEYWORDS:
        if kw in api:
            return False

    # Default: assume read-only (conservative for speculation safety —
    # unknown actions won't mutate env if they're actually read-only)
    return True


def is_mutating(action_name: str) -> bool:
    return not is_read_only(action_name)
