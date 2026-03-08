"""Intelligent entity deduplication using similarity-based merging.

Replaces naive exact-name dedup with Jaccard similarity scoring,
synonym expansion, and configurable thresholds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.common.logger import get_logger
from src.data_processing.entity_extractor import Entity


@dataclass
class MergeConfig:
    """Configuration for entity merging behavior."""

    similarity_threshold: float = 0.75
    name_weight: float = 0.5
    description_weight: float = 0.3
    attribute_weight: float = 0.2
    # Custom synonym groups: each inner list is a group of equivalent names
    synonym_groups: list[list[str]] = field(default_factory=list)


class EntityMerger:
    """Merge duplicate/similar entities using weighted similarity scoring.

    Scoring formula:
      score = name_weight * name_sim + description_weight * desc_sim
            + attribute_weight * attr_sim

    Where name_sim uses Jaccard similarity on character n-grams plus
    exact alias matching.
    """

    def __init__(self, config: MergeConfig | None = None) -> None:
        self._config = config or MergeConfig()
        self._logger = get_logger(__name__)
        self._synonym_map: dict[str, str] = self._build_synonym_map()

    # ── Public API ───────────────────────────────────────────────

    def merge(self, entities: list[Entity]) -> list[Entity]:
        """Merge similar entities, returning deduplicated list."""
        if not entities:
            return []

        # Phase 1: normalize and group by canonical name
        canonical_groups: dict[str, list[Entity]] = {}
        for entity in entities:
            canonical = self._canonicalize(entity.name)
            canonical_groups.setdefault(canonical, []).append(entity)

        # Phase 2: merge within each canonical group
        merged_list: list[Entity] = []
        for group in canonical_groups.values():
            merged_list.append(self._merge_group(group))

        # Phase 3: cross-group similarity merge
        final = self._cross_group_merge(merged_list)

        removed = len(entities) - len(final)
        if removed > 0:
            self._logger.info(
                f"Entity merge: {len(entities)} → {len(final)} "
                f"({removed} duplicates merged)"
            )
        return final

    # ── Canonicalization ─────────────────────────────────────────

    def _canonicalize(self, name: str) -> str:
        """Reduce name to canonical form for grouping."""
        name = name.strip()
        # Check synonym map
        lowered = name.lower()
        if lowered in self._synonym_map:
            return self._synonym_map[lowered]
        return lowered

    def _build_synonym_map(self) -> dict[str, str]:
        """Build lowered-name → canonical mapping from synonym groups."""
        mapping: dict[str, str] = {}
        for group in self._config.synonym_groups:
            if not group:
                continue
            canonical = group[0].lower()
            for name in group:
                mapping[name.lower()] = canonical
        return mapping

    # ── Group merging ────────────────────────────────────────────

    @staticmethod
    def _merge_group(group: list[Entity]) -> Entity:
        """Merge a group of entities with the same canonical name."""
        if len(group) == 1:
            return group[0]

        # Use the first entity as base, preferring longest name
        base = max(group, key=lambda e: len(e.name))
        merged_aliases: list[str] = list(base.aliases)
        merged_props: dict[str, object] = dict(base.properties)

        for entity in group:
            if entity is base:
                continue
            # Collect name as alias if different
            if entity.name != base.name and entity.name not in merged_aliases:
                merged_aliases.append(entity.name)
            # Collect aliases
            for alias in entity.aliases:
                if alias not in merged_aliases and alias != base.name:
                    merged_aliases.append(alias)
            # Merge properties (first-wins)
            for key, value in entity.properties.items():
                merged_props.setdefault(key, value)

        return Entity(
            name=base.name,
            entity_type=base.entity_type,
            aliases=merged_aliases,
            properties=merged_props,
        )

    # ── Cross-group similarity merge ─────────────────────────────

    def _cross_group_merge(self, entities: list[Entity]) -> list[Entity]:
        """Merge entities across groups based on similarity threshold."""
        if len(entities) <= 1:
            return entities

        # Track which entities have been merged into another
        merged_into: dict[int, int] = {}  # index → target index
        result_entities = list(entities)

        for i in range(len(result_entities)):
            if i in merged_into:
                continue
            for j in range(i + 1, len(result_entities)):
                if j in merged_into:
                    continue
                sim = self._compute_similarity(result_entities[i], result_entities[j])
                if sim >= self._config.similarity_threshold:
                    self._logger.debug(
                        f"Merging '{result_entities[j].name}' into "
                        f"'{result_entities[i].name}' (similarity={sim:.3f})"
                    )
                    result_entities[i] = self._merge_group(
                        [result_entities[i], result_entities[j]]
                    )
                    merged_into[j] = i

        return [e for idx, e in enumerate(result_entities) if idx not in merged_into]

    # ── Similarity computation ───────────────────────────────────

    def _compute_similarity(self, a: Entity, b: Entity) -> float:
        """Compute weighted similarity between two entities."""
        cfg = self._config

        name_sim = self._name_similarity(a, b)
        desc_sim = self._description_similarity(a, b)
        attr_sim = self._attribute_similarity(a, b)

        return (
            cfg.name_weight * name_sim
            + cfg.description_weight * desc_sim
            + cfg.attribute_weight * attr_sim
        )

    def _name_similarity(self, a: Entity, b: Entity) -> float:
        """Compute name similarity including aliases and synonyms."""
        # Exact match
        if a.name.lower() == b.name.lower():
            return 1.0

        # Alias overlap: check if any name in a's set matches b's set
        a_names = {a.name.lower()} | {alias.lower() for alias in a.aliases}
        b_names = {b.name.lower()} | {alias.lower() for alias in b.aliases}
        if a_names & b_names:
            return 1.0

        # Synonym map match
        a_canonical = self._canonicalize(a.name)
        b_canonical = self._canonicalize(b.name)
        if a_canonical == b_canonical:
            return 1.0

        # Character-level Jaccard similarity (bigrams)
        return self._jaccard_bigram(a.name, b.name)

    @staticmethod
    def _description_similarity(a: Entity, b: Entity) -> float:
        """Compute description similarity via entity_type matching."""
        if not a.entity_type or not b.entity_type:
            return 0.0
        if a.entity_type.lower() == b.entity_type.lower():
            return 1.0
        # Partial: check substring containment
        a_type = a.entity_type.lower()
        b_type = b.entity_type.lower()
        if a_type in b_type or b_type in a_type:
            return 0.5
        return 0.0

    @staticmethod
    def _attribute_similarity(a: Entity, b: Entity) -> float:
        """Compute similarity of properties dictionaries."""
        if not a.properties and not b.properties:
            return 0.0
        if not a.properties or not b.properties:
            return 0.0
        a_keys = set(a.properties.keys())
        b_keys = set(b.properties.keys())
        if not a_keys and not b_keys:
            return 0.0
        intersection = a_keys & b_keys
        union = a_keys | b_keys
        if not union:
            return 0.0
        return len(intersection) / len(union)

    @staticmethod
    def _jaccard_bigram(s1: str, s2: str) -> float:
        """Character bigram Jaccard similarity.

        Works well for both CJK and Latin text since it operates
        on character pairs rather than word boundaries.
        """
        # Normalize: strip whitespace and lowercase
        s1 = re.sub(r"\s+", "", s1.lower())
        s2 = re.sub(r"\s+", "", s2.lower())

        if not s1 or not s2:
            return 0.0

        if len(s1) < 2 or len(s2) < 2:
            # For single-char strings, fall back to exact match
            return 1.0 if s1 == s2 else 0.0

        bigrams_1 = {s1[i : i + 2] for i in range(len(s1) - 1)}
        bigrams_2 = {s2[i : i + 2] for i in range(len(s2) - 1)}

        intersection = bigrams_1 & bigrams_2
        union = bigrams_1 | bigrams_2

        if not union:
            return 0.0
        return len(intersection) / len(union)
