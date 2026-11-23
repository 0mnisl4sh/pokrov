"""Declarative policy evaluation engine for Pokrov."""

from __future__ import annotations

from typing import Literal

from pokrov.core.compliance_map import get_compliance_refs
from pokrov.core.config import CategoryRule, PokrovConfig
from pokrov.core.contracts import (
    Action,
    ComplianceRef,
    EntityCategory,
    InspectionResult,
    PolicyDecision,
    max_action,
)


class DeclarativePolicyEngine:
    """Evaluates inspection findings against declarative Pokrov policy profiles.

    Implements the ``PolicyEngine`` protocol.
    """

    def __init__(self, config: PokrovConfig) -> None:
        self._config = config

    @property
    def config(self) -> PokrovConfig:
        """Return the underlying policy configuration."""
        return self._config

    def evaluate(
        self,
        inspection: InspectionResult,
        *,
        direction: Literal["input", "output"] = "input",
        principal: str = "anonymous",
        profile: str | None = None,
    ) -> PolicyDecision:
        """Evaluate detections against the selected profile and principal overrides."""
        effective_profile = profile or self._config.active_profile
        prof = self._config.get_profile(profile)
        override_rules = prof.principal_overrides.get(principal, {})

        matched_actions: list[Action] = []
        seen_categories: set[EntityCategory] = set()
        ordered_categories: list[EntityCategory] = []
        seen_rules: set[str] = set()
        ordered_rules: list[str] = []
        seen_refs: set[tuple[str, str]] = set()
        ordered_refs: list[ComplianceRef] = []

        for det in inspection.detections:
            rule, is_override = _lookup_rule(det.category, override_rules, prof.rules)
            if rule is None or not rule.enabled or det.confidence < rule.min_confidence:
                continue

            effective_action = (
                rule.output_action
                if direction == "output" and rule.output_action is not None
                else rule.action
            )
            matched_actions.append(effective_action)

            if det.category not in seen_categories:
                seen_categories.add(det.category)
                ordered_categories.append(det.category)

            rule_id = (
                f"{effective_profile}:override:{principal}:{det.category.value}="
                f"{effective_action.value}"
                if is_override
                else f"{effective_profile}:{det.category.value}={effective_action.value}"
            )
            if rule_id not in seen_rules:
                seen_rules.add(rule_id)
                ordered_rules.append(rule_id)

            for ref in (*det.compliance_refs, *get_compliance_refs(det.category)):
                ref_key = (ref.framework, ref.article)
                if ref_key not in seen_refs:
                    seen_refs.add(ref_key)
                    ordered_refs.append(ref)

        final_action = max_action(matched_actions) if matched_actions else prof.default_action
        target_upstream = (
            self._config.router.local_upstream if final_action == Action.ROUTE_LOCAL else None
        )
        http_status = 403 if final_action in {Action.BLOCK, Action.REQUIRE_APPROVAL} else 200

        if ordered_categories:
            cats_str = ", ".join(cat.value for cat in ordered_categories)
            reason = (
                f"Profile '{effective_profile}' ({direction}) resolved '{final_action.value}' "
                f"for categories: {cats_str}."
            )
        else:
            reason = (
                f"No matching rules in profile '{effective_profile}' ({direction}); "
                f"applied default action '{final_action.value}'."
            )

        return PolicyDecision(
            action=final_action,
            reason=reason,
            matched_rules=tuple(ordered_rules),
            target_upstream=target_upstream,
            categories=tuple(ordered_categories),
            compliance_refs=tuple(ordered_refs),
            http_status=http_status,
        )


def _lookup_rule(
    category: EntityCategory,
    override_rules: dict[EntityCategory, CategoryRule],
    profile_rules: dict[EntityCategory, CategoryRule],
) -> tuple[CategoryRule | None, bool]:
    """Return ``(rule, is_override)`` prioritizing principal overrides."""
    if category in override_rules:
        return override_rules[category], True
    return profile_rules.get(category), False


__all__ = ["DeclarativePolicyEngine"]
