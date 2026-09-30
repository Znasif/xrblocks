import * as THREE from 'three';

let nextId = 1;

/**
 * A persistent real-world scene graph: nodes are grounded detections, edges
 * are relations. Scans are merged rather than replaced, so a cup stays the
 * same node as the user walks, and the graph is what tools query.
 */
export class SceneGraph {
  constructor({mergeDistance = 0.45, edgeDecay = 0.85} = {}) {
    this.mergeDistance = mergeDistance;
    this.edgeDecay = edgeDecay;
    /** @type {Map<string, {id: string, label: string, point: THREE.Vector3|null, seen: number, lastScan: number}>} */
    this.nodes = new Map();
    /** @type {Map<string, {sub: string, obj: string, predicate: string, score: number, spatial: boolean, seen: number, lastScan: number}>} */
    this.edges = new Map();
    this.scanCount = 0;
  }

  /**
   * Merges one scan. `detections` are `{label, point}` in scan order and
   * `triplets` index into them.
   * @returns the node id for each detection index.
   */
  ingest(detections, triplets) {
    this.scanCount += 1;
    const scan = this.scanCount;
    const ids = detections.map((d) => this.upsertNode_(d, scan));

    // Edges not re-observed this scan decay so stale relations fade out.
    for (const edge of this.edges.values()) {
      if (edge.lastScan !== scan) edge.score *= this.edgeDecay;
    }
    for (const t of triplets) {
      const sub = ids[t.subject];
      const obj = ids[t.object];
      if (!sub || !obj || sub === obj) continue;
      const key = `${sub}|${t.predicate}|${obj}`;
      const existing = this.edges.get(key);
      if (existing) {
        existing.score = Math.max(
          existing.score * 0.5 + t.score * 0.5,
          t.score
        );
        existing.seen += 1;
        existing.lastScan = scan;
      } else {
        this.edges.set(key, {
          sub,
          obj,
          predicate: t.predicate,
          score: t.score,
          spatial: !!t.spatial,
          seen: 1,
          lastScan: scan,
        });
      }
    }
    for (const [key, edge] of this.edges) {
      if (edge.score < 0.1) this.edges.delete(key);
    }
    return ids;
  }

  upsertNode_(detection, scan) {
    const label = detection.label.toLowerCase().trim();
    let best = null;
    let bestDistance = this.mergeDistance;
    if (detection.point) {
      for (const node of this.nodes.values()) {
        if (node.label !== label || !node.point) continue;
        const distance = node.point.distanceTo(detection.point);
        if (distance < bestDistance) {
          best = node;
          bestDistance = distance;
        }
      }
    }
    if (best) {
      best.point.lerp(detection.point, 0.5);
      best.seen += 1;
      best.lastScan = scan;
      return best.id;
    }
    const id = `${label.replace(/\s+/g, '_')}_${nextId++}`;
    this.nodes.set(id, {
      id,
      label,
      point: detection.point ? detection.point.clone() : null,
      seen: 1,
      lastScan: scan,
    });
    return id;
  }

  /** Edges touching a node, strongest first. */
  relationsOf(nodeId) {
    return [...this.edges.values()]
      .filter((e) => e.sub === nodeId || e.obj === nodeId)
      .sort((a, b) => b.score - a.score);
  }

  /** Nodes whose label contains `label`. */
  find(label) {
    const needle = label.toLowerCase().trim();
    return [...this.nodes.values()].filter(
      (n) =>
        n.label === needle ||
        n.label.includes(needle) ||
        needle.includes(n.label)
    );
  }

  /** A spoken-style summary of one node, the shape a `where_is` tool returns. */
  describe(nodeId) {
    const node = this.nodes.get(nodeId);
    if (!node) return '';
    const parts = this.relationsOf(nodeId)
      .slice(0, 3)
      .map((e) =>
        e.sub === nodeId
          ? `${e.predicate} the ${this.nodes.get(e.obj)?.label}`
          : `the ${this.nodes.get(e.sub)?.label} is ${e.predicate} it`
      );
    return parts.length
      ? `${node.label}: ${parts.join(', ')}`
      : `${node.label}: no relations yet`;
  }

  toJSON() {
    return {
      nodes: [...this.nodes.values()].map((n) => ({
        id: n.id,
        label: n.label,
        point: n.point ? n.point.toArray().map((v) => +v.toFixed(3)) : null,
        seen: n.seen,
      })),
      edges: [...this.edges.values()].map((e) => ({
        ...e,
        score: +e.score.toFixed(3),
      })),
    };
  }
}
