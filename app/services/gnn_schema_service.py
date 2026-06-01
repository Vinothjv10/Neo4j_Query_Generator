"""
GNN-Inspired Schema Linking via Graph Structural Embeddings.

Builds a schema graph from Neo4j (Table nodes + DEPENDS_ON / sibling-via-MAPS_TO
edges) using NetworkX and computes per-table structural scores that augment
the existing TF-IDF + embedding retrieval.

Architecture
------------
1. build_graph()  — runs at startup; pulls all tables + relationships from Neo4j
2. Node features per table:
     tier_score      : t3=1.00, t2=0.70, t1=0.40 (domain tier importance)
     col_count_norm  : columns / max_columns (schema richness)
     degree_cent     : NetworkX degree centrality (hub importance)
     pagerank        : PageRank score (global importance)
     maps_to_count   : # outgoing MAPS_TO relationships (join richness)
3. 1-hop message passing (mean aggregation of neighbour features):
     structural_emb[t] = concat(own_features[t], mean(features[neighbours[t]]))
   — this is a single GNN layer without learnable weights
4. propagate_from_seeds(seed_tables) — given an initial retrieval result,
   walk the schema graph to compute reachability scores for every table
5. get_structural_boost(table_name) — returns a [0,1] score per table that
   can be added to the TF-IDF + embedding score in table_index_service

Upgrade path
------------
Once you have 200+ labelled (question → correct_tables) pairs, replace the
unsupervised structural scores with a trained GraphSAGE / GAT model via
PyG (torch-geometric).  The interface stays identical.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from app.utils.logger import log_step

if TYPE_CHECKING:
    from app.services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)


class GNNSchemaService:
    def __init__(self, neo4j_service: "Neo4jService") -> None:
        self._neo4j = neo4j_service
        self._graph: Any = None          # networkx.DiGraph once built
        self._node_features: dict[str, np.ndarray] = {}
        self._structural_embs: dict[str, np.ndarray] = {}
        self._pagerank: dict[str, float] = {}
        self._degree_cent: dict[str, float] = {}
        self._ready = False

    # ── Startup ──────────────────────────────────────────────────────────────

    async def build_graph(self) -> None:
        """
        Pull all tables + relationships from Neo4j and build the schema graph.
        Computes structural features and 1-hop message-passing embeddings.
        """
        try:
            import networkx as nx
        except ImportError:
            log_step("GNN", "networkx not installed — GNN schema service disabled")
            return

        log_step("GNN", "Building schema graph from Neo4j")

        tables_raw = await self._fetch_all_tables_with_relationships()
        if not tables_raw:
            log_step("GNN", "No tables returned from Neo4j")
            return

        G: Any = nx.DiGraph()

        # Add nodes
        for tbl in tables_raw:
            name = tbl["name"]
            G.add_node(name, **{
                "schema": tbl.get("schema", "silver_layer"),
                "description": tbl.get("description") or "",
                "col_count": len(tbl.get("columns", [])),
                "maps_to_count": tbl.get("maps_to_count", 0),
            })

        # Add edges: DEPENDS_ON (t3→t2, t2→t1)
        for tbl in tables_raw:
            for dep in tbl.get("depends_on", []):
                if G.has_node(dep):
                    G.add_edge(tbl["name"], dep, rel="DEPENDS_ON")

        # Add edges: sibling via MAPS_TO column overlap
        for tbl in tables_raw:
            for sibling in tbl.get("siblings", []):
                if G.has_node(sibling) and not G.has_edge(tbl["name"], sibling):
                    G.add_edge(tbl["name"], sibling, rel="MAPS_TO_SIBLING")

        self._graph = G

        # Compute graph-level centrality metrics
        self._pagerank = nx.pagerank(G, alpha=0.85)
        self._degree_cent = nx.degree_centrality(G)

        # Build per-node feature vectors
        max_cols = max((G.nodes[n]["col_count"] for n in G.nodes), default=1) or 1
        max_pr = max(self._pagerank.values()) if self._pagerank else 1.0
        max_deg = max(self._degree_cent.values()) if self._degree_cent else 1.0

        for node in G.nodes:
            name_lower = node.lower()
            tier_score = 1.0 if name_lower.startswith("t3_") else \
                         0.7 if name_lower.startswith("t2_") else 0.4
            col_norm = G.nodes[node]["col_count"] / max_cols
            maps_norm = min(G.nodes[node]["maps_to_count"] / 10.0, 1.0)
            pr_norm = self._pagerank.get(node, 0.0) / max_pr if max_pr > 0 else 0.0
            deg_norm = self._degree_cent.get(node, 0.0) / max_deg if max_deg > 0 else 0.0

            self._node_features[node] = np.array(
                [tier_score, col_norm, maps_norm, pr_norm, deg_norm],
                dtype=np.float32,
            )

        # 1-hop message passing (mean-aggregate neighbour features)
        feat_dim = 5
        for node in G.nodes:
            own = self._node_features[node]
            neighbours = list(G.predecessors(node)) + list(G.successors(node))
            if neighbours:
                nb_vecs = np.stack(
                    [self._node_features[n] for n in neighbours if n in self._node_features],
                    axis=0,
                )
                mean_nb = nb_vecs.mean(axis=0)
            else:
                mean_nb = np.zeros(feat_dim, dtype=np.float32)
            # Concat own + mean(neighbours) → 10-dim structural embedding
            self._structural_embs[node] = np.concatenate([own, mean_nb])

        self._ready = True
        log_step(
            "GNN",
            f"Schema graph built: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges",
        )

    # ── Query-time scoring ───────────────────────────────────────────────────

    def propagate_from_seeds(
        self,
        seed_tables: list[str],
        depth: int = 2,
        decay: float = 0.5,
    ) -> dict[str, float]:
        """
        Given seed tables (initial retrieval results), walk the schema graph
        and assign a reachability score to every reachable table.

        score[seed]         = 1.0
        score[1-hop away]   = decay      (default 0.5)
        score[2-hop away]   = decay^2    (default 0.25)

        These scores augment the existing retrieval scores in TableIndexService.
        """
        if not self._ready or not self._graph:
            return {}

        scores: dict[str, float] = {}
        frontier: dict[str, float] = {t: 1.0 for t in seed_tables if self._graph.has_node(t)}

        for d in range(depth + 1):
            for node, score in frontier.items():
                if node not in scores or scores[node] < score:
                    scores[node] = score
            if d == depth:
                break
            next_frontier: dict[str, float] = {}
            for node, score in frontier.items():
                for nb in (
                    list(self._graph.predecessors(node)) +
                    list(self._graph.successors(node))
                ):
                    nb_score = score * (decay ** (d + 1))
                    if nb not in scores or nb_score > scores.get(nb, 0.0):
                        next_frontier[nb] = max(next_frontier.get(nb, 0.0), nb_score)
            frontier = next_frontier

        return scores

    def get_structural_score(self, table_name: str) -> float:
        """
        Return the global structural importance score for a table [0, 1].
        Combines tier_score, PageRank, and degree centrality.
        """
        if not self._ready:
            return 0.0
        feat = self._node_features.get(table_name)
        if feat is None:
            return 0.0
        # feat = [tier_score, col_norm, maps_norm, pr_norm, deg_norm]
        # Weighted sum: tier dominates
        weights = np.array([0.40, 0.15, 0.15, 0.20, 0.10], dtype=np.float32)
        return float(np.dot(feat, weights))

    def rerank_candidates(
        self,
        candidates: list[str],
        seed_tables: list[str] | None = None,
        gnn_weight: float = 0.25,
        semantic_scores: dict[str, float] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Re-rank *candidates* by blending semantic_scores with GNN structural scores.

        gnn_weight controls the blend: 0.0 = pure semantic, 1.0 = pure GNN.
        """
        if not self._ready:
            return [(t, (semantic_scores or {}).get(t, 0.0)) for t in candidates]

        prop_scores = self.propagate_from_seeds(seed_tables or candidates)

        results: list[tuple[str, float]] = []
        for tbl in candidates:
            sem = (semantic_scores or {}).get(tbl, 0.0)
            gnn_struct = self.get_structural_score(tbl)
            gnn_prop = prop_scores.get(tbl, 0.0)
            gnn = 0.5 * gnn_struct + 0.5 * gnn_prop
            blended = (1.0 - gnn_weight) * sem + gnn_weight * gnn
            results.append((tbl, blended))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Neo4j fetch ──────────────────────────────────────────────────────────

    async def _fetch_all_tables_with_relationships(self) -> list[dict]:
        """
        Fetch all table nodes with:
          - DEPENDS_ON relationships
          - MAPS_TO sibling tables (via shared column mappings)
          - maps_to_count (number of MAPS_TO edges from this table's columns)
        """
        query = """
        MATCH (t:Table)
        OPTIONAL MATCH (t)-[:DEPENDS_ON]->(dep:Table)
        OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
        OPTIONAL MATCH (c)-[:MAPS_TO]->(oc:Column)<-[:HAS_COLUMN]-(sibling:Table)
          WHERE sibling.name <> t.name
        WITH t,
             collect(DISTINCT dep.name) AS depends_on,
             collect(DISTINCT sibling.name) AS siblings,
             count(DISTINCT oc) AS maps_to_count,
             collect(DISTINCT {name: c.name, data_type: c.data_type}) AS columns
        RETURN t.name AS name,
               COALESCE(t.schema, 'silver_layer') AS schema,
               t.description AS description,
               depends_on,
               siblings,
               maps_to_count,
               columns
        """
        async with self._neo4j._driver.session() as session:
            result = await session.run(query)
            records = await result.data()

        out = []
        for r in records:
            out.append({
                "name": r["name"],
                "schema": r.get("schema", "silver_layer"),
                "description": r.get("description") or "",
                "depends_on": [d for d in (r.get("depends_on") or []) if d],
                "siblings": [s for s in (r.get("siblings") or []) if s],
                "maps_to_count": int(r.get("maps_to_count") or 0),
                "columns": [c for c in (r.get("columns") or []) if c.get("name")],
            })
        return out
