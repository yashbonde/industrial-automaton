"""Graphs & geometry tasks.

Each generator returns {"input": np.ndarray, "output": np.ndarray,
"input_formatted": list[str], "output_formatted": list[str]}.

Graphs are encoded as edge triples: [u1, v1, w1, u2, v2, w2, ...]
with 1-indexed node IDs (V.PAD = padding). input length / 3 = max edges.
Geometry tasks use 2D point sets.
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, dijkstra, breadth_first_order

from .. import vocab as V


_MAX_ITEMS = 5_000


def _default_rng(rng):
    if rng is None:
        return np.random.default_rng()
    return rng


def _check_items(n, limit=_MAX_ITEMS, **ctx):
    if n > limit:
        detail = ", ".join(f"{k}={v}" for k, v in ctx.items())
        raise ValueError(f"num items {n} exceeds limit {limit} ({detail})")


# --------------- Graph helpers ---------------


def _generate_connected_graph(rng, num_nodes, max_weight=None):
    """Generate a random connected graph as an adjacency matrix."""
    perm = rng.permutation(num_nodes)
    u = perm[:-1]
    v = perm[1:]

    weighted = max_weight is not None
    if weighted:
        w = rng.integers(1, max_weight + 1, size=num_nodes - 1)
    else:
        w = np.ones(num_nodes - 1, dtype=np.int64)

    adj = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    adj[u, v] = w
    adj[v, u] = w

    num_extra = rng.integers(0, num_nodes)
    if num_extra > 0:
        extra_u = rng.integers(0, num_nodes, size=num_extra)
        extra_v = rng.integers(0, num_nodes, size=num_extra)
        mask = extra_u != extra_v
        if weighted:
            extra_w = rng.integers(1, max_weight + 1, size=num_extra)
        else:
            extra_w = np.ones(num_extra, dtype=np.int64)
        adj[extra_u[mask], extra_v[mask]] = extra_w[mask]
        adj[extra_v[mask], extra_u[mask]] = extra_w[mask]

    return adj


def _adj_to_edge_triples(adj):
    """Extract undirected edges as (u, v, w) triples. 1-indexed nodes."""
    rows, cols = np.nonzero(adj)
    mask = rows < cols
    rows = rows[mask] + 1
    cols = cols[mask] + 1
    weights = adj[rows - 1, cols - 1]
    return np.column_stack([rows, cols, weights])


def _edge_triples_to_flat(triples, length):
    """Flatten edge triples into a PAD-padded 1D array."""
    flat = np.full(length, V.PAD, dtype=np.int64)
    n = min(len(triples) * 3, length)
    if n > 0:
        flat[:n] = triples.flatten()[:n]
    return flat


def _fmt_edges(triples, weighted=True):
    if len(triples) == 0:
        return "{}"
    if weighted:
        edges = [f"{t[0]}-{t[1]}:{t[2]}" for t in triples]
    else:
        edges = [f"{t[0]}-{t[1]}" for t in triples]
    return "{" + ", ".join(edges) + "}"


# --------------- Geometry helpers ---------------


def _points_to_triples(coords):
    """Encode 2D points as (id, x, y) triples. 1-indexed IDs."""
    n = len(coords)
    ids = np.arange(1, n + 1).reshape(-1, 1)
    return np.hstack([ids, coords])


def _triples_to_flat(triples, length):
    """Flatten point triples into a PAD-padded 1D array."""
    flat = np.full(length, V.PAD, dtype=np.int64)
    n = min(len(triples) * 3, length)
    if n > 0:
        flat[:n] = triples.flatten()[:n]
    return flat


def _fmt_points(coords):
    return (
        "["
        + ", ".join(
            f"({int(coords[i, 0])},{int(coords[i, 1])})" for i in range(len(coords))
        )
        + "]"
    )


# --------------- Graph generators ---------------


def generate_shortest_path(
    rng, batch_size: int, length: int, *, num_nodes: int = 10, max_weight: int = 9
):
    """Shortest path between random src and tgt nodes."""
    rng = _default_rng(rng)
    if length % 3 != 0:
        raise ValueError("length must be divisible by 3 (edge triples)")
    _check_items(num_nodes, num_nodes=num_nodes)

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.full((batch_size, num_nodes), V.PAD, dtype=np.int64)

    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        adj = _generate_connected_graph(rng, num_nodes, max_weight=max_weight)

        num_extra = num_nodes
        u_extra = rng.integers(0, num_nodes, size=num_extra)
        v_extra = rng.integers(0, num_nodes, size=num_extra)
        mask = u_extra != v_extra
        w_extra = rng.integers(1, max_weight + 1, size=num_extra)
        adj[u_extra[mask], v_extra[mask]] = w_extra[mask]
        adj[v_extra[mask], u_extra[mask]] = w_extra[mask]

        triples = _adj_to_edge_triples(adj)

        src = rng.integers(0, num_nodes)
        tgt = rng.integers(0, num_nodes)
        while tgt == src:
            tgt = rng.integers(0, num_nodes)

        flat_edges = _edge_triples_to_flat(triples, length - 3)
        inputs[b, : length - 3] = flat_edges
        inputs[b, length - 3] = src + 1
        inputs[b, length - 2] = tgt + 1
        inputs[b, length - 1] = V.PAD

        dist_matrix_result, predecessors = dijkstra(
            adj, directed=False, indices=[src], return_predecessors=True
        )

        path = []
        curr = tgt
        if dist_matrix_result[0, tgt] != np.inf:
            path = [tgt]
            while curr != src:
                curr = predecessors[0, curr]
                path.append(curr)
            path.reverse()

        if len(path) > 0:
            out_path = np.array(path, dtype=np.int64) + 1
            n_out = min(len(out_path), num_nodes)
            outputs[b, :n_out] = out_path[:n_out]

        if num_nodes <= 20:
            in_fmt.append(f"graph {_fmt_edges(triples)}, find path {src}->{tgt}")
            path_str = " -> ".join(str(n) for n in path)
            out_fmt.append(f"path: {path_str}")
        else:
            in_fmt.append("graph (large)")
            out_fmt.append("path (large)")

    return {
        "input": inputs,
        "output": outputs,
        "input_formatted": in_fmt,
        "output_formatted": out_fmt,
    }


def generate_mst_prim(rng, batch_size: int, length: int, *, num_nodes: int = 10, max_weight: int = 9):
    """Minimum spanning tree of a weighted graph."""
    rng = _default_rng(rng)
    if length % 3 != 0:
        raise ValueError("length must be divisible by 3 (edge triples)")
    _check_items(num_nodes, num_nodes=num_nodes)

    mst_len = 3 * (num_nodes - 1)
    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.full((batch_size, mst_len), V.PAD, dtype=np.int64)

    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        adj = _generate_connected_graph(rng, num_nodes, max_weight=max_weight)
        triples = _adj_to_edge_triples(adj)
        inputs[b] = _edge_triples_to_flat(triples, length)

        mst_sparse = minimum_spanning_tree(adj)
        mst = mst_sparse.toarray().astype(np.int64)
        mst = np.maximum(mst, mst.T)
        mst_triples = _adj_to_edge_triples(mst)
        outputs[b] = _edge_triples_to_flat(mst_triples, mst_len)

        if num_nodes <= 20:
            in_fmt.append(f"graph {_fmt_edges(triples)}")
            total_w = sum(int(t[2]) for t in mst_triples)
            out_fmt.append(f"MST {_fmt_edges(mst_triples)} (weight={total_w})")
        else:
            in_fmt.append("graph (large)")
            out_fmt.append("MST (large)")

    return {
        "input": inputs,
        "output": outputs,
        "input_formatted": in_fmt,
        "output_formatted": out_fmt,
    }


def generate_graph_traversal(rng, batch_size: int, length: int, *, num_nodes: int = 10):
    """BFS traversal order on an unweighted graph."""
    rng = _default_rng(rng)
    if length % 3 != 0:
        raise ValueError("length must be divisible by 3 (edge triples)")
    _check_items(num_nodes, num_nodes=num_nodes)

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.zeros((batch_size, num_nodes), dtype=np.int64)

    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        adj = _generate_connected_graph(rng, num_nodes, max_weight=None)
        triples = _adj_to_edge_triples(adj)
        inputs[b] = _edge_triples_to_flat(triples, length)

        node_order, predecessors = breadth_first_order(adj, i_start=0, directed=False)

        rank = np.zeros(num_nodes, dtype=np.int64)
        rank[node_order] = np.arange(1, len(node_order) + 1)
        outputs[b] = rank

        if num_nodes <= 20:
            in_fmt.append(f"graph {_fmt_edges(triples, weighted=False)}, src=0")
            out_fmt.append(f"BFS order: {' -> '.join(str(n) for n in node_order)}")
        else:
            in_fmt.append("graph (large), src=0")
            out_fmt.append("BFS order (large)")

    return {
        "input": inputs,
        "output": outputs,
        "input_formatted": in_fmt,
        "output_formatted": out_fmt,
    }


# --------------- Geometry generators ---------------


def generate_tsp(rng, batch_size, length, *, num_cities=None, coord_scale=100):
    """TSP nearest neighbor heuristic.

    Input: point triples [id1,x1,y1, ...], 1-indexed IDs.
    Output: tour as 1-indexed city IDs, PAD-padded.
    """
    rng = _default_rng(rng)
    if length % 3 != 0:
        raise ValueError("length must be divisible by 3 (point triples)")
    if num_cities is None:
        num_cities = length // 3
    _check_items(num_cities, num_cities=num_cities)

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.full((batch_size, num_cities), V.PAD, dtype=np.int64)

    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        coords = rng.integers(0, coord_scale, size=(num_cities, 2))
        triples = _points_to_triples(coords)
        inputs[b] = _triples_to_flat(triples, length)

        dists = distance_matrix(coords, coords)

        current = 0
        tour = [0]
        dists[:, 0] = np.inf

        for _ in range(num_cities - 1):
            next_city = np.argmin(dists[current])
            tour.append(int(next_city))
            dists[:, next_city] = np.inf
            current = next_city

        outputs[b] = np.array(tour, dtype=np.int64) + 1

        if num_cities <= 20:
            in_fmt.append(f"cities {_fmt_points(coords)}")
            out_fmt.append(f"tour: {' -> '.join(str(c + 1) for c in tour)}")
        else:
            in_fmt.append("cities (large)")
            out_fmt.append("tour (large)")

    return {
        "input": inputs,
        "output": outputs,
        "input_formatted": in_fmt,
        "output_formatted": out_fmt,
    }


def generate_convex_hull(
    rng, batch_size, length, *, num_points=None, coord_scale=100
):
    """Convex Hull.

    Input: point triples [id1,x1,y1, ...], 1-indexed IDs.
    Output: binary mask — TRUE if on hull, FALSE otherwise.
    """
    rng = _default_rng(rng)
    if length % 3 != 0:
        raise ValueError("length must be divisible by 3 (point triples)")
    if num_points is None:
        num_points = length // 3
    _check_items(num_points, num_points=num_points)

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.full((batch_size, num_points), V.FALSE, dtype=np.int64)

    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        points = rng.integers(0, coord_scale, size=(num_points, 2))
        triples = _points_to_triples(points)
        inputs[b] = _triples_to_flat(triples, length)

        if num_points >= 3:
            try:
                hull = ConvexHull(points)
                outputs[b, hull.vertices] = V.TRUE
                hull_indices = sorted(hull.vertices)
            except Exception:
                hull_indices = []
        else:
            outputs[b, :] = V.TRUE
            hull_indices = list(range(num_points))

        if num_points <= 20:
            in_fmt.append(f"points {_fmt_points(points)}")
            hull_pts = [
                f"p{i + 1}({int(points[i, 0])},{int(points[i, 1])})"
                for i in hull_indices
            ]
            out_fmt.append(f"hull: {{{', '.join(hull_pts)}}}")
        else:
            in_fmt.append("points (large)")
            out_fmt.append("hull (large)")

    return {
        "input": inputs,
        "output": outputs,
        "input_formatted": in_fmt,
        "output_formatted": out_fmt,
    }


def generate_delaunay(
    rng,
    batch_size,
    length,
    *,
    num_points=None,
    coord_scale=100,
    max_triangles=None,
):
    """Delaunay triangulation.

    Input: point triples [id1,x1,y1, ...], 1-indexed IDs.
    Output: triangle vertex triples (1-indexed), PAD-padded.
    """
    rng = _default_rng(rng)
    if length % 3 != 0:
        raise ValueError("length must be divisible by 3 (point triples)")
    if num_points is None:
        num_points = length // 3
    _check_items(num_points, num_points=num_points)
    if max_triangles is None:
        max_triangles = max(1, 2 * num_points - 5)

    inputs = np.full((batch_size, length), V.PAD, dtype=np.int64)
    outputs = np.full((batch_size, max_triangles * 3), V.PAD, dtype=np.int64)

    in_fmt = []
    out_fmt = []

    for b in range(batch_size):
        points = rng.integers(0, coord_scale, size=(num_points, 2))
        triples = _points_to_triples(points)
        inputs[b] = _triples_to_flat(triples, length)

        triangles = []
        if num_points >= 3:
            try:
                delaunay = Delaunay(points)
                triangles = delaunay.simplices
            except Exception:
                pass

        n_tris = min(len(triangles), max_triangles)
        if n_tris > 0:
            outputs[b, : n_tris * 3] = np.array(triangles[:n_tris]).flatten() + 1

        if num_points <= 20:
            in_fmt.append(f"points {_fmt_points(points)}")
            tri_strs = [
                f"({t[0] + 1},{t[1] + 1},{t[2] + 1})" for t in triangles[:n_tris]
            ]
            out_fmt.append(f"triangles: [{', '.join(tri_strs)}]")
        else:
            in_fmt.append("points (large)")
            out_fmt.append("triangles (large)")

    return {
        "input": inputs,
        "output": outputs,
        "input_formatted": in_fmt,
        "output_formatted": out_fmt,
    }


# ---- Vocab info functions ----

def _shortest_path_vocab_info(**_kw):
    toks = set(range(0, 100)) | {V.PAD}
    out = set(range(0, 100)) | {V.PAD}
    return {"input_tokens": toks, "output_tokens": out, "output_mask": V._make_mask(out)}


def _mst_prim_vocab_info(**_kw):
    toks = set(range(0, 100)) | {V.PAD}
    return {"input_tokens": toks, "output_tokens": toks, "output_mask": V._make_mask(toks)}


def _graph_traversal_vocab_info(**_kw):
    toks = set(range(0, 100)) | {V.PAD}
    return {"input_tokens": toks, "output_tokens": toks, "output_mask": V._make_mask(toks)}


def _tsp_vocab_info(**_kw):
    toks = set(range(0, 100)) | {V.PAD}
    return {"input_tokens": toks, "output_tokens": toks, "output_mask": V._make_mask(toks)}


def _convex_hull_vocab_info(**_kw):
    toks = set(range(0, 100)) | {V.PAD}
    out = {V.TRUE, V.FALSE}
    return {"input_tokens": toks, "output_tokens": out, "output_mask": V._make_mask(out)}


def _delaunay_vocab_info(**_kw):
    toks = set(range(0, 100)) | {V.PAD}
    return {"input_tokens": toks, "output_tokens": toks, "output_mask": V._make_mask(toks)}


VOCAB_INFO = {
    "shortest_path": _shortest_path_vocab_info,
    "mst_prim": _mst_prim_vocab_info,
    "graph_traversal": _graph_traversal_vocab_info,
    "tsp": _tsp_vocab_info,
    "convex_hull": _convex_hull_vocab_info,
    "delaunay": _delaunay_vocab_info,
}
