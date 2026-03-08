"""
GraphService — in-memory NetworkX DiGraph over document relationships.

Thread-safety:
  All mutations are guarded by self._lock (asyncio.Lock).
  All public methods that touch self.graph are async and acquire the lock.
  Router handlers must be async def to use this service.

Key fixes vs original:
  - asyncio.Lock on every read/write (was unprotected)
  - BFS uses collections.deque — O(1) popleft instead of O(n) list.pop(0)
  - max_nodes cap on traverse() prevents unbounded memory use
  - Edge weight incremented (not reset) when an edge already exists
  - Orphan detection uses both in_degree==0 AND out_degree==0 (true orphan)
  - get_stats() returns node_count / edge_count / orphan_count / hub_nodes
"""

import asyncio
import logging
from collections import deque
from typing import Any, Dict, List, Optional, Set

import networkx as nx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from models.orm import Document, Link

logger = logging.getLogger(__name__)


class GraphService:
    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._lock = asyncio.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def build_from_db(self, session: AsyncSession) -> None:
        """
        Clears and rebuilds the entire in-memory graph from the database.
        Called once at startup — not per-request.
        Fetches all Documents and all Links in exactly two queries.
        """
        async with self._lock:
            self.graph.clear()

            # One query for all nodes
            doc_result = await session.execute(select(Document))
            for doc in doc_result.scalars().all():
                self.graph.add_node(doc.id, title=doc.title)

            # One query for all edges
            link_result = await session.execute(select(Link))
            for link in link_result.scalars().all():
                self.graph.add_edge(
                    link.source_doc_id,
                    link.target_doc_id,
                    weight=link.weight,
                )

        logger.info(
            "GraphService: built from DB — %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    # ── Mutations ──────────────────────────────────────────────────────────────

    async def update_document_links(
        self,
        doc_id: str,
        doc_title: str,
        old_targets: Set[str],
        new_targets_info: Dict[str, str],
    ) -> None:
        """
        Incrementally diffs and updates edges for a single document.

        new_targets_info: {target_id → target_title}
        Increments weight on existing edges instead of resetting to 1.
        """
        async with self._lock:
            # Ensure the source node exists
            if doc_id not in self.graph:
                self.graph.add_node(doc_id, title=doc_title)
            else:
                self.graph.nodes[doc_id]["title"] = doc_title

            new_targets = set(new_targets_info.keys())

            # Remove edges that no longer exist
            for target in old_targets - new_targets:
                if self.graph.has_edge(doc_id, target):
                    self.graph.remove_edge(doc_id, target)

            # Add new edges or increment weight of existing ones
            for target in new_targets - old_targets:
                if target not in self.graph:
                    self.graph.add_node(target, title=new_targets_info[target])
                if self.graph.has_edge(doc_id, target):
                    self.graph[doc_id][target]["weight"] += 1
                else:
                    self.graph.add_edge(doc_id, target, weight=1)

    async def add_document(self, doc_id: str, title: str) -> None:
        """Adds a single document node (called on document creation)."""
        async with self._lock:
            if doc_id not in self.graph:
                self.graph.add_node(doc_id, title=title)

    async def delete_document(self, doc_id: str) -> None:
        """Removes a document node and all its incident edges."""
        async with self._lock:
            if doc_id in self.graph:
                self.graph.remove_node(doc_id)
                logger.debug("GraphService: removed node %s", doc_id)

    # ── Queries ────────────────────────────────────────────────────────────────

    async def traverse(
        self,
        start_id: str,
        depth: int = 2,
        max_nodes: int = 500,
    ) -> Dict[str, Any]:
        """
        BFS from start_id up to `depth` hops.

        Returns:
            {"nodes": [...], "truncated": bool}

        Uses deque for O(1) popleft. Stops immediately when max_nodes is
        reached to prevent unbounded memory use on large graphs.
        """
        async with self._lock:
            if start_id not in self.graph:
                return {"nodes": [], "truncated": False}

            results: List[Dict[str, Any]] = []
            visited: Set[str] = {start_id}
            queue: deque = deque([(start_id, 0)])
            truncated = False

            while queue:
                current_id, current_depth = queue.popleft()

                results.append({
                    "id": current_id,
                    "depth": current_depth,
                    "title": self.graph.nodes[current_id].get("title", "Untitled"),
                })

                if len(results) >= max_nodes:
                    truncated = True
                    break

                if current_depth < depth:
                    for neighbor in self.graph.neighbors(current_id):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, current_depth + 1))

        return {"nodes": results, "truncated": truncated}

    async def get_all_nodes_edges(self) -> Dict[str, List[Dict[str, Any]]]:
        """Full graph dump for visualization — nodes and edges with weights."""
        async with self._lock:
            nodes = [
                {"id": n, "title": data.get("title", "Untitled")}
                for n, data in self.graph.nodes(data=True)
            ]
            edges = [
                {"source": u, "target": v, "weight": data.get("weight", 1)}
                for u, v, data in self.graph.edges(data=True)
            ]
        return {"nodes": nodes, "edges": edges}

    async def get_stats(self) -> Dict[str, Any]:
        """
        Returns aggregate graph metrics.

        orphan_count: nodes with BOTH in_degree==0 AND out_degree==0.
        hub_nodes: top 5 by in_degree.
        """
        async with self._lock:
            total_nodes = self.graph.number_of_nodes()
            total_edges = self.graph.number_of_edges()

            orphans = [
                n for n in self.graph.nodes()
                if self.graph.in_degree(n) == 0 and self.graph.out_degree(n) == 0
            ]

            in_degrees = sorted(
                self.graph.in_degree(), key=lambda x: x[1], reverse=True
            )
            hub_nodes = [
                {
                    "id": n,
                    "title": self.graph.nodes[n].get("title", "Untitled"),
                    "in_degree": d,
                }
                for n, d in in_degrees[:5]
            ]

        return {
            "node_count": total_nodes,
            "edge_count": total_edges,
            "orphan_count": len(orphans),
            "hub_nodes": hub_nodes,
        }


# Singleton — imported by routers and services
graph_service = GraphService()
