#!/usr/bin/env python3
"""Graph construction and solver engine for hard SML reasoning evaluation.

Builds complex graphs with verified answers. All solver methods compute
ground-truth answers that the question generator uses to build
guaranteed-correct multiple choice questions.

Usage:
    # Self-test
    python sml_opaque_eval/graph_engine.py
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict, deque
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── SML formatting helpers (reuse existing conventions) ──────────────────────


def E(idx: int, conf: float = 0.9) -> str:
    """Entity descriptor line."""
    return f"E(0|0|0|0|X{idx}|0|0|{conf})"


def R(rtype: str, src: int, tgt: int, weight: float = 0.85) -> str:
    """Relation line."""
    return f"R({rtype}|{src}|{tgt}|{weight:.2f}|0|0)"


def sml_block(ents: list[str], rels: list[str]) -> str:
    """Wrap entity + relation lines in <sml> block."""
    return "<sml>\n" + "\n".join(ents + rels) + "\n</sml>"


# ── Graph class ──────────────────────────────────────────────────────────────


class Graph:
    """Directed weighted graph with SML serialization and solver methods."""

    def __init__(self):
        self.entities: list[dict] = []  # [{idx, conf, properties}]
        self.relations: list[dict] = []  # [{rel_type, src, tgt, weight}]
        self._adj: dict[int, list[dict]] = defaultdict(list)  # outgoing
        self._radj: dict[int, list[dict]] = defaultdict(list)  # incoming

    # ── Construction ─────────────────────────────────────────────────────

    def add_entity(self, conf: float = 0.9, properties: list[str] | None = None) -> int:
        """Add an entity, returns its index."""
        idx = len(self.entities)
        self.entities.append({
            "idx": idx,
            "conf": conf,
            "properties": list(properties or []),
        })
        return idx

    def add_relation(self, rel_type: str, src: int, tgt: int, weight: float = 0.85) -> None:
        """Add a directed weighted relation."""
        rel = {"rel_type": rel_type, "src": src, "tgt": tgt, "weight": weight}
        self.relations.append(rel)
        self._adj[src].append(rel)
        self._radj[tgt].append(rel)

    def add_chain(
        self,
        rel_types: list[str],
        weights: list[float] | None = None,
        conf: float = 0.9,
        start_idx: int | None = None,
    ) -> list[int]:
        """Build a multi-hop chain, returns list of node indices.

        If start_idx is given, the chain begins at that existing node.
        Otherwise, a new start node is created.
        """
        if weights is None:
            weights = [0.85] * len(rel_types)
        assert len(rel_types) == len(weights)

        if start_idx is not None:
            nodes = [start_idx]
        else:
            nodes = [self.add_entity(conf)]

        for rt, w in zip(rel_types, weights):
            new = self.add_entity(conf)
            nodes.append(new)
            self.add_relation(rt, nodes[-2], new, w)

        return nodes

    def add_distractor_branch(
        self,
        from_node: int,
        depth: int,
        rel_types: list[str],
        weights: list[float] | None = None,
        conf: float = 0.9,
    ) -> list[int]:
        """Add a dead-end branch from an existing node. Returns new node indices."""
        if weights is None:
            weights = [round(random.uniform(0.4, 0.9), 2)] * depth
        branch_nodes = [from_node]
        for i in range(depth):
            rt = rel_types[i % len(rel_types)]
            w = weights[i] if i < len(weights) else 0.7
            new = self.add_entity(conf)
            branch_nodes.append(new)
            self.add_relation(rt, branch_nodes[-2], new, w)
        return branch_nodes[1:]  # exclude the from_node

    def to_sml(self) -> str:
        """Serialize to SML block."""
        ents = [E(e["idx"], e["conf"]) for e in self.entities]
        rels = [R(r["rel_type"], r["src"], r["tgt"], r["weight"]) for r in self.relations]
        return sml_block(ents, rels)

    def entity_name(self, idx: int) -> str:
        """Return entity display name."""
        return f"X{idx}"

    def entity_names(self, indices: list[int] | set[int]) -> list[str]:
        """Return sorted entity display names."""
        return [f"X{i}" for i in sorted(indices)]

    # ── Solver methods ───────────────────────────────────────────────────

    def reachable_from(
        self, start: int, rel_filter: str | set[str] | None = None
    ) -> set[int]:
        """BFS reachability from start, optionally filtered by relation type(s)."""
        if isinstance(rel_filter, str):
            rel_filter = {rel_filter}
        visited = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for rel in self._adj.get(node, []):
                if rel_filter and rel["rel_type"] not in rel_filter:
                    continue
                if rel["tgt"] not in visited:
                    queue.append(rel["tgt"])
        visited.discard(start)
        return visited

    def chain_endpoint(self, start: int, rel_types: list[str]) -> int | None:
        """Follow specific relation types in sequence to terminus.

        Returns the final node reached, or None if the chain breaks.
        """
        current = start
        for rt in rel_types:
            found = None
            for rel in self._adj.get(current, []):
                if rel["rel_type"] == rt:
                    found = rel["tgt"]
                    break
            if found is None:
                return None
            current = found
        return current

    def all_paths(
        self, start: int, end: int, max_depth: int = 10
    ) -> list[list[int]]:
        """Enumerate all simple paths from start to end up to max_depth."""
        paths = []

        def _dfs(node: int, path: list[int]):
            if len(path) > max_depth + 1:
                return
            if node == end:
                paths.append(list(path))
                return
            for rel in self._adj.get(node, []):
                if rel["tgt"] not in path:
                    path.append(rel["tgt"])
                    _dfs(rel["tgt"], path)
                    path.pop()

        _dfs(start, [start])
        return paths

    def effective_properties(self, entity: int) -> set[str]:
        """Compute inherited properties minus NOT_ overrides.

        Walks IsA chains to accumulate properties from ancestors,
        then removes any negated via NOT_HasProperty relations.
        """
        # Collect all ancestors via IsA
        ancestors = set()
        queue = deque([entity])
        while queue:
            node = queue.popleft()
            for rel in self._adj.get(node, []):
                if rel["rel_type"] == "IsA" and rel["tgt"] not in ancestors:
                    ancestors.add(rel["tgt"])
                    queue.append(rel["tgt"])

        # Gather properties from entity and all ancestors
        props = set(self.entities[entity]["properties"])
        for anc in ancestors:
            props.update(self.entities[anc]["properties"])

        # Gather via HasProperty relations
        for node in {entity} | ancestors:
            for rel in self._adj.get(node, []):
                if rel["rel_type"] == "HasProperty":
                    props.add(self.entity_name(rel["tgt"]))

        # Remove NOT_ overrides (entity-level NOT_HasProperty)
        negated = set()
        for rel in self._adj.get(entity, []):
            if rel["rel_type"] == "NOT_HasProperty":
                negated.add(self.entity_name(rel["tgt"]))

        return props - negated

    def prerequisite_chain_satisfiable(self, entity: int) -> tuple[bool, list[int]]:
        """Check if the prerequisite chain from entity can be completed.

        Returns (satisfiable, blocking_nodes) where blocking_nodes are
        entities that are required but blocked by NOT_ relations.
        """
        # Follow HasPrerequisite chain
        prereqs = set()
        queue = deque([entity])
        visited = {entity}
        while queue:
            node = queue.popleft()
            for rel in self._adj.get(node, []):
                if rel["rel_type"] == "HasPrerequisite" and rel["tgt"] not in visited:
                    prereqs.add(rel["tgt"])
                    visited.add(rel["tgt"])
                    queue.append(rel["tgt"])

        # Check for NOT_CapableOf blocking any prerequisite
        blocked = set()
        for rel in self._adj.get(entity, []):
            if rel["rel_type"] == "NOT_CapableOf" and rel["tgt"] in prereqs:
                blocked.add(rel["tgt"])

        return len(blocked) == 0, sorted(blocked)

    def confidence_filtered_reachable(
        self,
        start: int,
        threshold: float,
        rel_filter: str | set[str] | None = None,
    ) -> set[int]:
        """BFS reachability only through relations with weight >= threshold."""
        if isinstance(rel_filter, str):
            rel_filter = {rel_filter}
        visited = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for rel in self._adj.get(node, []):
                if rel["weight"] < threshold:
                    continue
                if rel_filter and rel["rel_type"] not in rel_filter:
                    continue
                if rel["tgt"] not in visited:
                    queue.append(rel["tgt"])
        visited.discard(start)
        return visited

    def hub_entities(self, top_k: int = 1) -> list[tuple[int, int]]:
        """Return top-k entities by total degree (in + out), as (idx, degree)."""
        degrees = defaultdict(int)
        for rel in self.relations:
            degrees[rel["src"]] += 1
            degrees[rel["tgt"]] += 1
        # Ensure all entities appear
        for e in self.entities:
            degrees.setdefault(e["idx"], 0)
        ranked = sorted(degrees.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:top_k]

    def isolated_entities(self) -> set[int]:
        """Return entities with zero connections."""
        connected = set()
        for rel in self.relations:
            connected.add(rel["src"])
            connected.add(rel["tgt"])
        return {e["idx"] for e in self.entities} - connected

    def degree(self, entity: int) -> int:
        """Total degree (in + out) for an entity."""
        count = 0
        for rel in self.relations:
            if rel["src"] == entity:
                count += 1
            if rel["tgt"] == entity:
                count += 1
        return count

    def out_degree(self, entity: int) -> int:
        """Out-degree for an entity."""
        return len(self._adj.get(entity, []))

    def in_degree(self, entity: int) -> int:
        """In-degree for an entity."""
        return len(self._radj.get(entity, []))

    def contradiction_paths(
        self, start: int, target: int
    ) -> tuple[list[dict], list[dict]]:
        """Find positive and negative paths from start to target.

        Returns (positive_rels, negative_rels) where positive_rels are
        non-NOT_ relations reaching target, and negative_rels are NOT_
        relations reaching target.
        """
        positive = []
        negative = []
        for rel in self.relations:
            if rel["src"] == start and rel["tgt"] == target:
                if rel["rel_type"].startswith("NOT_"):
                    negative.append(rel)
                else:
                    positive.append(rel)
        return positive, negative

    def strongest_path(
        self, start: int, target: int, max_depth: int = 10
    ) -> tuple[float, list[int]] | None:
        """Find the path with highest minimum-weight edge (bottleneck path).

        Returns (min_weight, path) or None if no path exists.
        """
        paths = self.all_paths(start, target, max_depth)
        if not paths:
            return None

        best_path = None
        best_min_weight = -1.0

        for path in paths:
            min_w = float("inf")
            for i in range(len(path) - 1):
                # Find the relation between consecutive nodes
                for rel in self._adj.get(path[i], []):
                    if rel["tgt"] == path[i + 1]:
                        min_w = min(min_w, rel["weight"])
                        break
            if min_w > best_min_weight:
                best_min_weight = min_w
                best_path = path

        if best_path is None:
            return None
        return best_min_weight, best_path

    def path_weight(self, path: list[int]) -> float:
        """Compute the minimum weight along a path (bottleneck)."""
        min_w = float("inf")
        for i in range(len(path) - 1):
            for rel in self._adj.get(path[i], []):
                if rel["tgt"] == path[i + 1]:
                    min_w = min(min_w, rel["weight"])
                    break
        return min_w if min_w != float("inf") else 0.0

    def cumulative_weight(self, path: list[int]) -> float:
        """Compute the product of weights along a path."""
        product = 1.0
        for i in range(len(path) - 1):
            for rel in self._adj.get(path[i], []):
                if rel["tgt"] == path[i + 1]:
                    product *= rel["weight"]
                    break
        return product

    def strongest_path_cumulative(
        self, start: int, target: int, max_depth: int = 10
    ) -> tuple[float, list[int]] | None:
        """Find the path with highest cumulative (product) weight."""
        paths = self.all_paths(start, target, max_depth)
        if not paths:
            return None

        best_path = None
        best_weight = -1.0

        for path in paths:
            w = self.cumulative_weight(path)
            if w > best_weight:
                best_weight = w
                best_path = path

        if best_path is None:
            return None
        return best_weight, best_path

    def relations_from(self, entity: int, rel_type: str | None = None) -> list[dict]:
        """Get all outgoing relations from an entity, optionally filtered."""
        rels = self._adj.get(entity, [])
        if rel_type:
            return [r for r in rels if r["rel_type"] == rel_type]
        return list(rels)

    def relations_to(self, entity: int, rel_type: str | None = None) -> list[dict]:
        """Get all incoming relations to an entity, optionally filtered."""
        rels = self._radj.get(entity, [])
        if rel_type:
            return [r for r in rels if r["rel_type"] == rel_type]
        return list(rels)

    def num_entities(self) -> int:
        return len(self.entities)

    def num_relations(self) -> int:
        return len(self.relations)


# ── Self-test ────────────────────────────────────────────────────────────────


def self_test():
    """Build known graphs and assert solver correctness."""
    passed = 0
    failed = 0

    def check(name: str, actual, expected):
        nonlocal passed, failed
        if actual == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name}")
            print(f"    expected: {expected}")
            print(f"    actual:   {actual}")

    # ── Test 1: Simple chain reachability ─────────────────────────────
    print("Test 1: Chain reachability")
    g = Graph()
    nodes = g.add_chain(["IsA", "CapableOf", "Causes"], [0.9, 0.8, 0.7])
    check("chain nodes", len(nodes), 4)
    check("reachable from 0", g.reachable_from(0), {1, 2, 3})
    check("reachable from 2", g.reachable_from(2), {3})
    check("reachable from 3", g.reachable_from(3), set())

    # ── Test 2: Chain endpoint ────────────────────────────────────────
    print("Test 2: Chain endpoint")
    check("chain_endpoint full", g.chain_endpoint(0, ["IsA", "CapableOf", "Causes"]), 3)
    check("chain_endpoint partial", g.chain_endpoint(0, ["IsA", "CapableOf"]), 2)
    check("chain_endpoint broken", g.chain_endpoint(0, ["IsA", "Causes"]), None)

    # ── Test 3: Filtered reachability ─────────────────────────────────
    print("Test 3: Filtered reachability")
    check("filtered IsA", g.reachable_from(0, "IsA"), {1})
    check("filtered Causes", g.reachable_from(0, "Causes"), set())
    check("filtered set", g.reachable_from(0, {"IsA", "CapableOf"}), {1, 2})

    # ── Test 4: Confidence threshold ──────────────────────────────────
    print("Test 4: Confidence threshold")
    check("threshold 0.75", g.confidence_filtered_reachable(0, 0.75), {1, 2})
    check("threshold 0.85", g.confidence_filtered_reachable(0, 0.85), {1})
    check("threshold 0.95", g.confidence_filtered_reachable(0, 0.95), set())

    # ── Test 5: Hub / degree / isolated ───────────────────────────────
    print("Test 5: Structural metrics")
    g2 = Graph()
    hub = g2.add_entity()  # 0
    for _ in range(4):
        n = g2.add_entity()
        g2.add_relation("RelatedTo", hub, n, 0.8)
    iso = g2.add_entity()  # 5

    check("hub", g2.hub_entities(1), [(0, 4)])
    check("isolated", g2.isolated_entities(), {5})
    check("degree hub", g2.degree(0), 4)
    check("degree iso", g2.degree(5), 0)

    # ── Test 6: All paths ─────────────────────────────────────────────
    print("Test 6: All paths")
    g3 = Graph()
    for _ in range(4):
        g3.add_entity()
    g3.add_relation("R1", 0, 1, 0.9)
    g3.add_relation("R2", 0, 2, 0.8)
    g3.add_relation("R3", 1, 3, 0.7)
    g3.add_relation("R4", 2, 3, 0.6)
    paths = g3.all_paths(0, 3)
    check("num paths 0→3", len(paths), 2)
    check("path via 1", [0, 1, 3] in paths, True)
    check("path via 2", [0, 2, 3] in paths, True)

    # ── Test 7: Strongest path ────────────────────────────────────────
    print("Test 7: Strongest path")
    result = g3.strongest_path(0, 3)
    check("strongest path exists", result is not None, True)
    if result:
        w, p = result
        check("strongest path route", p, [0, 1, 3])
        check("strongest path weight", w, 0.7)

    # ── Test 8: Effective properties with NOT_ override ───────────────
    print("Test 8: Property inheritance + negation")
    g4 = Graph()
    child = g4.add_entity(properties=["fast"])
    parent = g4.add_entity(properties=["strong", "durable"])
    prop_node = g4.add_entity()  # X2 — used as property target
    neg_prop = g4.add_entity()   # X3 — negated property target
    g4.add_relation("IsA", child, parent, 0.9)
    g4.add_relation("HasProperty", parent, prop_node, 0.8)
    g4.add_relation("NOT_HasProperty", child, prop_node, 0.9)

    props = g4.effective_properties(child)
    check("has inherited strong", "strong" in props, True)
    check("has own fast", "fast" in props, True)
    check("NOT_ override removed X2", "X2" not in props, True)

    # ── Test 9: Prerequisite satisfiability ───────────────────────────
    print("Test 9: Prerequisite chain")
    g5 = Graph()
    for _ in range(4):
        g5.add_entity()
    g5.add_relation("HasPrerequisite", 0, 1, 0.9)
    g5.add_relation("HasPrerequisite", 1, 2, 0.8)
    g5.add_relation("NOT_CapableOf", 0, 2, 0.95)

    sat, blocked = g5.prerequisite_chain_satisfiable(0)
    check("blocked", sat, False)
    check("blocked nodes", blocked, [2])

    # ── Test 10: Contradiction paths ──────────────────────────────────
    print("Test 10: Contradiction paths")
    g6 = Graph()
    for _ in range(3):
        g6.add_entity()
    g6.add_relation("CapableOf", 0, 1, 0.9)
    g6.add_relation("NOT_CapableOf", 0, 1, 0.8)
    pos, neg = g6.contradiction_paths(0, 1)
    check("positive paths", len(pos), 1)
    check("negative paths", len(neg), 1)

    # ── Test 11: SML output ───────────────────────────────────────────
    print("Test 11: SML serialization")
    g7 = Graph()
    g7.add_entity(0.9)
    g7.add_entity(0.8)
    g7.add_relation("IsA", 0, 1, 0.85)
    sml_out = g7.to_sml()
    check("sml has <sml>", sml_out.startswith("<sml>"), True)
    check("sml has </sml>", sml_out.endswith("</sml>"), True)
    check("sml has E(0", "E(0|0|0|0|X0|0|0|0.9)" in sml_out, True)
    check("sml has R(", "R(IsA|0|1|0.85|0|0)" in sml_out, True)

    # ── Test 12: Cumulative path weight ───────────────────────────────
    print("Test 12: Cumulative path weight")
    result = g3.strongest_path_cumulative(0, 3)
    check("cumulative path exists", result is not None, True)
    if result:
        w, p = result
        # path via 1: 0.9 * 0.7 = 0.63, path via 2: 0.8 * 0.6 = 0.48
        check("cumulative best path", p, [0, 1, 3])
        check("cumulative weight", round(w, 4), round(0.9 * 0.7, 4))

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Self-test: {passed} passed, {failed} failed")
    if failed > 0:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    self_test()
