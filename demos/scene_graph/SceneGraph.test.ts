import * as THREE from 'three';
import {describe, expect, it} from 'vitest';

import {SceneGraph} from './SceneGraph.js';

const det = (label: string, x: number, y: number, z: number) => ({
  label,
  point: new THREE.Vector3(x, y, z),
});

describe('SceneGraph', () => {
  it('creates nodes and edges from one scan', () => {
    const graph = new SceneGraph();
    const ids = graph.ingest(
      [det('cup', 0, 0.8, -1), det('table', 0, 0.7, -1)],
      [{subject: 0, object: 1, predicate: 'on', score: 0.9, spatial: false}]
    );
    expect(ids).toHaveLength(2);
    expect(graph.nodes.size).toBe(2);
    expect(graph.edges.size).toBe(1);
    const edge = [...graph.edges.values()][0];
    expect(edge.sub).toBe(ids[0]);
    expect(edge.obj).toBe(ids[1]);
    expect(graph.describe(ids[0])).toBe('cup: on the table');
  });

  it('merges a re-observed object into the same node', () => {
    const graph = new SceneGraph({mergeDistance: 0.45});
    const [cupA] = graph.ingest([det('cup', 0, 0.8, -1)], []);
    const [cupB] = graph.ingest([det('cup', 0.1, 0.8, -1.1)], []);
    expect(cupB).toBe(cupA);
    expect(graph.nodes.size).toBe(1);
    expect(graph.nodes.get(cupA)!.seen).toBe(2);
  });

  it('keeps same-label objects apart when they are far away', () => {
    const graph = new SceneGraph({mergeDistance: 0.45});
    graph.ingest([det('chair', 0, 0.5, -1), det('chair', 2, 0.5, -1)], []);
    expect(graph.nodes.size).toBe(2);
  });

  it('never merges across different labels', () => {
    const graph = new SceneGraph();
    graph.ingest([det('cup', 0, 0.8, -1)], []);
    graph.ingest([det('bottle', 0, 0.8, -1)], []);
    expect(graph.nodes.size).toBe(2);
  });

  it('decays edges that stop being observed and drops them eventually', () => {
    const graph = new SceneGraph({edgeDecay: 0.5});
    const scan = [det('cup', 0, 0.8, -1), det('table', 0, 0.7, -1)];
    graph.ingest(scan, [
      {subject: 0, object: 1, predicate: 'on', score: 0.8, spatial: false},
    ]);
    graph.ingest(scan, []);
    expect([...graph.edges.values()][0].score).toBeCloseTo(0.4);
    graph.ingest(scan, []);
    graph.ingest(scan, []);
    graph.ingest(scan, []);
    expect(graph.edges.size).toBe(0);
  });

  it('ignores self relations and out-of-range indices', () => {
    const graph = new SceneGraph();
    graph.ingest(
      [det('cup', 0, 0.8, -1)],
      [
        {subject: 0, object: 0, predicate: 'on', score: 0.9, spatial: false},
        {subject: 0, object: 5, predicate: 'on', score: 0.9, spatial: false},
      ]
    );
    expect(graph.edges.size).toBe(0);
  });

  it('finds nodes by partial label and serializes', () => {
    const graph = new SceneGraph();
    graph.ingest([det('potted plant', 1, 0.5, -2)], []);
    expect(graph.find('plant')).toHaveLength(1);
    const json = graph.toJSON();
    expect(json.nodes[0].label).toBe('potted plant');
    expect(json.nodes[0].point).toEqual([1, 0.5, -2]);
  });
});
