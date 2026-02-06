import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import math
import subprocess
import xml.etree.ElementTree as ET
import json
import os
from plotly.colors import qualitative
from plotly.colors import sample_colorscale
from collections import deque
from typing import Literal
from copy import copy, deepcopy
from collections import Counter

# Constants
MAX_RAD = 2
RESOLUTION = 20
EARTH_RADIUS = 6371
MTBF = {
    "OXC": 10**5,
    "mux": 1.67 * 10**5,
    "fiber": 2.63 * 10**6,
    "OA": 2.5 * 10**5,
    "transponder": 3.5 * 10**5,
}


# Geometric and Graph functions-------------------------------------------------


def get_endpoints_by_id(G: nx.Graph, edge_id: int) -> tuple:
    """
    Finds and returns the endpoints (nodes) of an edge in a graph by its edge ID.
    Args:
        G (networkx.Graph): The graph containing the edges.
        edge_id (Any): The unique identifier of the edge to search for.
    Returns:
        tuple: A tuple (u, v) representing the endpoints of the edge if found.
    Raises:
        AssertionError: If the edge with the specified ID is not found in the graph.
    """

    for u, v, data in G.edges(data=True):
        if data["id"] == edge_id:
            return u, v
    assert False, f"Edge with id {edge_id} not found in the graph."


def planar_embed(G: nx.Graph) -> nx.PlanarEmbedding:
    """
    Creates a planar embedding of a graph.
    Parameters:
        G (nx.Graph): The input graph to be embedded.
    Returns:
        nx.PlanarEmbedding: A planar embedding of the input graph G, retaining all node and edge data from G.
    The function performs the following steps:
    1. Initializes an empty planar embedding.
    2. Iterates through the nodes of the graph in depth-first search (DFS) preorder.
    3. For each node, determines its neighbors in clockwise order based on the provided positions.
    4. Adds half-edges to the embedding, ensuring the correct clockwise or counterclockwise order.
    5. Checks the structure of the embedding.
    6. Copies node and edge data from the input graph G to the embedding.
    Note:
    - The function assumes that the input graph G is planar.
    - The function uses a helper function `neighbors_in_cw` to determine the neighbors in clockwise order.
    - The input graph G must have a 'pos' attribute for each node, which is used to determine the clockwise order of neighbors.
    """
    embedding = nx.PlanarEmbedding()
    nx.dfs_preorder_nodes(G)
    for node in nx.dfs_preorder_nodes(G):
        neighbors = list(G.neighbors(node))
        neighbors.sort(
            key=lambda neighbor: math.atan2(
                G.nodes[neighbor]["pos"][0] - G.nodes[node]["pos"][0],
                G.nodes[neighbor]["pos"][1] - G.nodes[node]["pos"][1],
            )
        )

        # If the node not exists yet, we can add it with all of its neighbors
        # This is only called once per component
        if node not in embedding:
            embedding.add_half_edge(node, neighbors[0])
            for i, neighbor in enumerate(neighbors[1:]):
                embedding.add_half_edge(node, neighbor, ccw=neighbors[i])

        # If the node exists, find a neighbor of it
        else:
            first_neighbor_index = next(
                i
                for i, neighbor in enumerate(neighbors)
                if (neighbor, node) in embedding.edges()
            )
            for i in range(first_neighbor_index, first_neighbor_index + len(neighbors)):
                neighbor = neighbors[i % len(neighbors)]
                # if there are no edges from node to neighbors
                if not list(embedding.neighbors(node)):
                    embedding.add_half_edge(node, neighbor)
                # if there is an edge from node to a neighbor
                else:
                    embedding.add_half_edge(
                        node, neighbor, ccw=neighbors[i % len(neighbors) - 1]
                    )
    embedding.check_structure()

    # Add node and edge data from G to the embedding
    nx.set_node_attributes(embedding, {n: d for n, d in G.nodes(data=True)})
    nx.set_edge_attributes(embedding, {(u, v): d for u, v, d in G.edges(data=True)})
    nx.set_edge_attributes(embedding, {(v, u): d for u, v, d in G.edges(data=True)})
    embedding.graph = G.graph.copy()

    return embedding


def get_faces(G: nx.PlanarEmbedding) -> list[list[int]]:
    """
    Get the faces of a planar embedding.
    Parameters:
        G (nx.PlanarEmbedding): A planar embedding of a graph.
    Returns:
        list: A list of faces, where each face is represented as a list of nodes in clockwise order.
    """
    faces = []
    traversed_edges = set()
    faces.extend(
        G.traverse_face(*edge, traversed_edges)
        for edge in G.edges()
        if edge not in traversed_edges
    )
    return faces


def get_outer_face(G: nx.PlanarEmbedding) -> list[int]:
    """
    Returns the outer face of a planar embedding.
    The outer face is the face that contains the most nodes and forms the boundary of the embedding.
    Parameters:
        G (nx.PlanarEmbedding): A planar embedding of a graph.
    Returns:
        list: A list of nodes that form the outer face of the planar embedding.
    """
    faces = get_faces(G)
    return max(faces, key=len)


def get_center_of_face(G: nx.Graph, face: list[int]) -> tuple:
    """
    Calculate the geometric center of a face given the positions of its nodes.

    Args:
        G (nx.Graph): A NetworkX graph with node positions stored in the "pos" attribute.
        face (list[int]): A list of node identifiers representing the face

    Returns:
        tuple: A tuple (x, y) representing the coordinates of the center of the face.
    """
    x = sum(G.nodes[node]["pos"][0] for node in face) / len(face)
    y = sum(G.nodes[node]["pos"][1] for node in face) / len(face)
    return x, y


def get_dividing_point_of_edge(
    G: nx.Graph, edge: tuple[int], section: float = 1 / 2
) -> tuple:
    """
    Returns the dividing point of an edge in a planar embedding.
    Parameters:
        G (nx.Graph): A NetworkX graph with node positions stored in the "pos" attribute.
        edge (tuple[int]): A tuple representing the edge (u, v).
        section (float): A float between 0 and 1 representing the section of the edge to divide.
                        Default is 1/2, which means the midpoint.
    Returns:
        tuple: A tuple representing the (x, y) coordinates of the dividing point on the edge.
    """
    u, v = edge
    x = G.nodes[u]["pos"][0] + (G.nodes[v]["pos"][0] - G.nodes[u]["pos"][0]) * section
    y = G.nodes[u]["pos"][1] + (G.nodes[v]["pos"][1] - G.nodes[u]["pos"][1]) * section
    return x, y


def quad_bezier(p0, ctrl, p2, resolution):
    """Return num points on a quadratic Bézier curve defined by points p0, p1, p2."""
    t_vals = [i / (resolution - 1) for i in range(resolution)]
    x_vals = [
        (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * ctrl[0] + t**2 * p2[0] for t in t_vals
    ]
    y_vals = [
        (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * ctrl[1] + t**2 * p2[1] for t in t_vals
    ]
    return x_vals, y_vals


def point_segment_distance(px, py, x1, y1, x2, y2):
    """
    Calculate the shortest distance from a point to a line segment.
    Parameters:
        px (float): x-coordinate of the point.
        py (float): y-coordinate of the point.
        x1 (float): x-coordinate of the first endpoint of the segment.
        y1 (float): y-coordinate of the first endpoint of the segment.
        x2 (float): x-coordinate of the second endpoint of the segment.
        y2 (float): y-coordinate of the second endpoint of the segment.
    Returns:
        float: The shortest distance from the point to the line segment.
    """

    # Convert to numpy arrays
    p = np.array([px, py])
    a = np.array([x1, y1])
    b = np.array([x2, y2])

    # Vector AB and AP
    ab = b - a
    ap = p - a

    # Project point P onto line AB, clamping to segment
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0, 1)  # Clamp to segment

    # Find the closest point on the segment
    closest = a + t * ab

    # Return the Euclidean distance
    return np.linalg.norm(p - closest)


def make_walks_by_edges(G: nx.Graph, walks_by_nodes: list) -> list:
    """
    Converts a list of walks represented by node sequences into a list of walks represented by edge sequences.
    Args:
        G (nx.Graph): The input graph.
        walks_by_nodes (list): A list of walks, where each walk is represented as a list of node identifiers.
    Returns:
        list: A list of walks, where each walk is represented as a list of edge identifiers.
    """
    walks_by_edges = []
    for walk in walks_by_nodes:
        edges = []
        for i in range(len(walk) - 1):
            u, v = walk[i], walk[i + 1]
            edge_id = G[u][v]["id"]
            edges.append(edge_id)
        walks_by_edges.append(edges)
    return walks_by_edges


def lower_convex_hull(
    points: list[tuple[float, float, int]],
) -> list[tuple[float, float, int]]:
    """
    Computes the lower convex hull of a set of 2D points.
    Points must be (x, y, id) tuples. Returns the lower hull as a list of points.
    """
    # Sort the points lexicographically by x then y
    points = sorted(points)

    lower = []
    for p in points:
        while len(lower) >= 2:
            (x1, y1, id1), (x2, y2, id2) = lower[-2], lower[-1]
            (x3, y3, id3) = p
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    return lower


def choose_random_nodes_by_distance(
    G: nx.Graph, distance: int, k: int = 10, seed: int = 42
) -> list[tuple[int, int]]:
    """
    Randomly selects `k` pairs of nodes from the graph `G` such that the distance between each pair is exactly `distance`.
    Args:
        G (nx.Graph): The input graph.
        distance (int): The exact distance between the selected node pairs.
        k (int): The number of node pairs to select.
    Returns:
        list[tuple[int, int]]: A list of `k` tuples, each containing a pair of node IDs.
    """
    # Compute all pairs shortest path lengths
    np.random.seed(seed)
    pairs = []
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    nodes = list(G.nodes)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            if lengths[u].get(v, -1) == distance:
                pairs.append((u, v))
    # Randomly select k pairs (without replacement)
    if len(pairs) < k:
        raise ValueError(
            f"Not enough node pairs at distance {distance} to select {k} pairs. There are only {len(pairs)} pairs available."
        )
    selected = list(np.random.choice(len(pairs), k, replace=False))
    return [pairs[i] for i in selected]


def remove_high_cost_edges(
    G: nx.Graph, s: int, t: int, cost_limit: int, remove_nodes: bool = False
):
    H = deepcopy(G)
    # Compute shortest path lengths from s and from t
    dist_s = nx.single_source_shortest_path_length(H, s)
    dist_t = nx.single_source_shortest_path_length(H, t)

    edges_to_remove = []

    for u, v in H.edges():
        cost = min(dist_s[u] + 1 + dist_t[v], dist_s[v] + 1 + dist_t[u])

        if cost > cost_limit:
            edges_to_remove.append((u, v))

    H.remove_edges_from(edges_to_remove)
    if remove_nodes:
        H.remove_nodes_from(list(nx.isolates(H)))
    edge_ids = {edge[2]["id"] for edge in H.edges(data=True)}

    # Now remake the SRLGs
    new_srlgs = deepcopy(H.graph["srlgs"])
    for srlg in new_srlgs:
        srlg["edges"] = list(set(srlg["edges"]) & edge_ids)
        srlg["valid"] = True if srlg["edges"] else False

    for i in range(len(new_srlgs) - 1):
        for j in range(i + 1, len(new_srlgs)):
            if set(new_srlgs[i]["edges"]) == set(new_srlgs[j]["edges"]):
                new_srlgs[i]["probability"] += new_srlgs[j]["probability"]
                new_srlgs[j]["valid"] = False

    new_srlgs = [srlg for srlg in new_srlgs if srlg["valid"]]
    for srlg in new_srlgs:
        srlg.pop("valid", None)

    H.graph["srlgs"] = new_srlgs

    return H


def great_circle_distance(u, v, radius=EARTH_RADIUS):
    lon1_deg, lat1_deg = u["pos"]
    lon2_deg, lat2_deg = v["pos"]

    # fok -> radián
    lat1 = math.radians(lat1_deg)
    lon1 = math.radians(lon1_deg)
    lat2 = math.radians(lat2_deg)
    lon2 = math.radians(lon2_deg)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # haversine formula
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.asin(math.sqrt(a))

    return radius * c


# Input generation -------------------------------------------------------------


def randomize_points(
    xs: list[float], ys: list[float], n: int
) -> list[tuple[float, float]]:
    """
    Randomly selects `n` points from the given lists of x and y coordinates.
    Each point is formed by pairing a randomly chosen x from `xs` and a randomly chosen y from `ys`.
    The resulting coordinates are rounded to two decimal places.
    Args:
        xs (list[float]): List of x coordinates.
        ys (list[float]): List of y coordinates.
        n (int): Number of random points to generate.
    Returns:
        list[tuple[float, float]]: List of `n` (x, y) coordinate tuples.
    Note:
        The selection of x and y coordinates is independent, so the returned points may not correspond to original (x, y) pairs.
    """

    random_xs = np.random.choice(xs, n)
    random_ys = np.random.choice(ys, n)
    coordinates = [(round(x, 2), round(y, 2)) for x, y in zip(random_xs, random_ys)]
    return coordinates


def generate_disk_srlgs(
    G: nx.Graph,
    coordinates: list[tuple[float, float]],
    r1: float,
    r2: float,
    r3: float,
) -> list[dict]:
    """
    Generates Shared Risk Link Groups (SRLGs) for a given graph based on disk-shaped regions centered at specified coordinates.
    For each coordinate, the function determines which edges of the graph fall within three concentric disks of radii r1, r2, and r3.
    Each SRLG is defined by a radius, the set of edge IDs within the corresponding disk, the center coordinate, and a multiplicity count
    indicating how many times the same SRLG configuration was encountered.
    Args:
        G (nx.Graph): The input graph. Each node must have a "pos" attribute (tuple of x, y coordinates).
                        Each edge must have an "id" attribute (unique identifier).
        coordinates (list[tuple[float, float]]): List of (x, y) tuples representing the centers of the disks.
        r1 (float): Radius of the smallest disk.
        r2 (float): Radius of the middle disk.
        r3 (float): Radius of the largest disk.
    Returns:
        list[dict]: A list of SRLG dictionaries, each containing:
            - "edges": Tuple of edge IDs within the disk.
            - "radius": The radius of the disk.
            - "center": The (x, y) coordinate of the disk center.
            - "multiplicity": The number of times this SRLG configuration was found.
    """

    srlgs = {}
    # (radius, edges) -> {center, multiplicity}
    # e.g., (3, (1, 3, 5)) -> {"center": (1, 5), "multiplicity": 5}

    for px, py in coordinates:
        r1_edges = []
        r2_edges = []
        r3_edges = []

        for edge in G.edges(data=True):
            # Get coordinates of the two endpoints of the edge
            x1, y1 = G.nodes[edge[0]]["pos"][0], G.nodes[edge[0]]["pos"][1]
            x2, y2 = G.nodes[edge[1]]["pos"][0], G.nodes[edge[1]]["pos"][1]

            # Calculate distance between the point and the edge
            dist = point_segment_distance(px, py, x1, y1, x2, y2)

            # If the distance is less than the radius, add the edge to the SRLG
            if dist <= r1:
                r1_edges.append(edge[2]["id"])
                r2_edges.append(edge[2]["id"])
                r3_edges.append(edge[2]["id"])
            elif dist <= r2:
                r2_edges.append(edge[2]["id"])
                r3_edges.append(edge[2]["id"])
            elif dist <= r3:
                r3_edges.append(edge[2]["id"])

        # Update the SRLGs dictionary with the edges and center
        srlgs.setdefault(
            (r1, frozenset(r1_edges)), {"center": (px, py), "multiplicity": 0}
        )["multiplicity"] += 1
        srlgs.setdefault(
            (r2, frozenset(r2_edges)), {"center": (px, py), "multiplicity": 0}
        )["multiplicity"] += 1
        srlgs.setdefault(
            (r3, frozenset(r3_edges)), {"center": (px, py), "multiplicity": 0}
        )["multiplicity"] += 1

    # Convert the dictionary to a list of SRLGs
    srlgs = [
        {
            "edges": tuple(edges),
            "radius": radius,
            "center": data["center"],
            "multiplicity": data["multiplicity"],
        }
        for (radius, edges), data in srlgs.items()
        if edges
    ]

    return srlgs


# SRLG functions----------------------------------------------------------------


def calculate_edge_rate(u, v):
    dist = great_circle_distance(u, v)
    k = dist // 100
    r = (
        365
        * 24
        * (
            dist / MTBF["fiber"]
            + k / MTBF["OA"]
            + 2 / MTBF["mux"]
            + 2 / MTBF["transponder"]
        )
    )
    return r


def redefine_srlgs(G):
    H = deepcopy(G)
    srlgs = G.graph["srlgs"]
    new_srlgs = []

    total_rate = 0
    for edge in G.edges():
        total_rate += calculate_edge_rate(G.nodes[edge[0]], G.nodes[edge[1]])
    total_rate += len(G.nodes()) * 365 * 24 / 10**5

    incident_edge_sets = set()  # frozensetek halmaza

    for v in G.nodes():
        incident_ids = {data["id"] for _, _, data in G.edges(v, data=True)}
        incident_edge_sets.add(frozenset(incident_ids))

    for srlg in srlgs:
        edge_set = frozenset(srlg["edges"])
        new_srlg = deepcopy(srlg)
        earthquake_rate = G.graph["rate"] * srlg["probability"]
        if len(edge_set) == 1 or edge_set in incident_edge_sets:  # edge or node SRLG
            if len(srlg["edges"]) == 1:  # edge
                edge_id = srlg["edges"][0]
                edge = G.graph["edge_by_id"][edge_id]
                device_rate = calculate_edge_rate(G.nodes[edge[0]], G.nodes[edge[1]])
                new_srlg["probability"] = (device_rate + earthquake_rate) / (
                    G.graph["rate"] + total_rate
                )
                new_srlgs.append(new_srlg)
            else:  # node
                new_srlg["probability"] = (365 * 24 / 10**5 + earthquake_rate) / (
                    G.graph["rate"] + total_rate
                )
                new_srlgs.append(new_srlg)
        else:  # other SRLG
            new_srlg["probability"] = earthquake_rate / (G.graph["rate"] + total_rate)
            new_srlgs.append(new_srlg)
    H.graph["srlgs"] = new_srlgs
    return H


def count_intersecting_srlgs(G: nx.Graph, srlgs: list[dict]) -> dict:
    """
    Count the number of SRLGs (Shared Risk Link Groups) that contain each edge in a graph.
    Parameters:
        G (nx.Graph): A NetworkX graph (also works with planar embeddings).
    Returns:
        dict: A dictionary where keys are edges (tuples of nodes) and values are the count of SRLGs that contain each edge.
    """
    # We initialize a dictionary to store how many SRLGs contain each edge.
    edge_count = {edge: 0 for edge in G.edges() if edge[0] > edge[1]}
    # Iterate through each SLRG and update the count for each edge
    for srlg in srlgs:
        for edge in srlg["edges"]:
            edge_count[
                next(
                    (u, v)
                    for u, v, d in G.edges(data=True)
                    if d["id"] == edge and u > v
                )
            ] += 1

    return edge_count


def remove_cutting_srlgs(G: nx.Graph, s: int, t: int) -> None:
    """
    Removes SRLGs (Shared Risk Link Groups) from the graph that, if their edges are removed, would disconnect nodes s and t.
    This function iterates over all SRLGs in the graph and removes those whose edge removals would disconnect the specified source and target nodes. The remaining SRLGs are those that do not affect the connectivity between s and t.
    Args:
        G (nx.Graph): The input NetworkX graph. The graph is expected to have a "srlgs" attribute (list of SRLGs) and an "edge_by_id" mapping in its graph attribute.
        s (int): The source node identifier.
        t (int): The target node identifier.
    Returns:
        None: The function modifies the graph in-place, updating the "srlgs" attribute.
    """

    cleaned_srlgs = []
    for srlg in G.graph["srlgs"]:
        # Create a copy of the graph to avoid modifying the original
        G_copy = G.copy()
        # Remove the edges in the current SRLG
        for edge_id in srlg["edges"]:
            u, v = G.graph["edge_by_id"][edge_id]
            G_copy.remove_edge(u, v)
        # Check if s_id and t_id are still connected
        if nx.has_path(G_copy, s, t):
            cleaned_srlgs.append(srlg)

    G.graph["srlgs"] = cleaned_srlgs


def has_free_edge(G: nx.Graph) -> bool:
    # Initialize a dictionary to store the count of SLRGs containing each edge
    edges = [edge[2]["id"] for edge in G.edges(data=True)]
    edge_count = {edge: 0 for edge in edges}

    # Iterate through each SLRG and update the count for each edge
    for srlg in G.graph["srlgs"]:
        for edge in srlg["edges"]:
            edge_count[edge] += 1

    # Check if there is an edge with a value of 0 in edge_count
    return any(count == 0 for count in edge_count.values())


def count_srlgs_intersecting_walks(G: nx.Graph, walks: list[list[int]]):
    """
    Counts how many SRLGs (Shared Risk Link Groups) intersect with each walk in the given graph.
    For each possible number of intersecting walks (from 0 up to the number of walks), this function
    returns a list of SRLGs that intersect with exactly that many walks.
    Args:
        G (nx.Graph): The input graph. The graph must have a "srlgs" attribute in G.graph, which is a list of SRLGs.
                      Each SRLG is expected to be a dictionary with an "edges" key, containing a list of edges.
        walks (list[list[int]]): A list of walks, where each walk is represented as a list of node IDs.
    Returns:
        list[list[dict]]: A list of lists, where the i-th list contains SRLGs that intersect with exactly i walks.
    """

    walks_by_edges = make_walks_by_edges(G, walks)
    # Count how many SRLGs intersect with each walk
    srlgs_intersecting_k_walks = [[] for _ in range(len(walks_by_edges) + 1)]
    for srlg in G.graph["srlgs"]:
        # Count how many walks this SRLG intersects
        count = sum(
            any(edge in walk for edge in srlg["edges"]) for walk in walks_by_edges
        )
        srlgs_intersecting_k_walks[count].append(srlg)
    return srlgs_intersecting_k_walks


def get_surviving_probabilities_of_walks(
    G: nx.Graph, walks: list[list[int]]
) -> list[float]:
    """
    Calculates the surviving probabilities for a set of walks in a graph based on SRLG (Shared Risk Link Group) failure probabilities.
    Args:
        G (nx.Graph): The input graph. The graph must have a "srlgs" attribute in G.graph, which is a list of dictionaries, each containing a "probability" key.
        walks (list[list[int]]): A list of walks, where each walk is represented as a list of node indices.
    Returns:
        list[float]: A list of surviving probabilities for the walks, where each entry corresponds to the probability that at least k walks survive, for k from len(walks) down to 1.
    Notes:
        - The function assumes the existence of a helper function `count_srlgs_intersecting_walks` that returns, for each k, the SRLGs intersecting with exactly k walks.
        - The surviving probability for k walks is computed as the sum of failure probabilities for all cases where at least k walks survive.
    """
    total_failure_probability = sum(srlg["probability"] for srlg in G.graph["srlgs"])
    failure_probabilities = []

    # Count how many SRLGs intersect with each walk
    srlgs_intersecting_k_walks = count_srlgs_intersecting_walks(G, walks)

    # calculate the exact failure probabilities for each k
    for srlgs in srlgs_intersecting_k_walks:
        failure_probabilities.append(sum(srlg["probability"] for srlg in srlgs))
    # add the probability of no SRLGs failing
    failure_probabilities[0] += 1 - total_failure_probability

    # Calculate the surviving probabilities
    surviving_probabilities = [
        sum(failure_probabilities[:k]) for k in range(len(failure_probabilities), 0, -1)
    ]

    return surviving_probabilities


def get_common_srlgs_of_walks(
    G: nx.graph, walks: list[list[int]], srlgs: list[dict]
) -> list[dict]:
    """
    Finds and returns the list of SRLGs (Shared Risk Link Groups) that intersect with all given walks in a graph.
    Args:
        G (nx.graph): The input graph.
        walks (list[list[int]]): A list of walks, where each walk is represented as a list of node IDs.
        srlgs (list[dict]): A list of SRLGs, where each SRLG is a dictionary containing at least the key "edges",
            which is a list of edges (tuples or other edge representations).
    Returns:
        list[dict]: A list of SRLG dictionaries that have at least one edge present in every walk.
    """
    common_srlgs = []
    walks_by_edges = make_walks_by_edges(G, walks)

    for srlg in srlgs:
        if all(any(edge in walk for edge in srlg["edges"]) for walk in walks_by_edges):
            common_srlgs.append(srlg)

    return common_srlgs


def shorten_paths(
    G: nx.Graph, s, t, og_paths: list[list[int]], srlgs: list[dict]
) -> list[list[int]]:
    """
    Attempts to iteratively shorten the given list of paths between nodes `s` and `t` in the graph `G`
    by replacing longer paths with shorter alternatives, while respecting SRLG (Shared Risk Link Group) constraints.
    The function works as follows:
    - Sorts the input paths by length in descending order.
    - For each path (starting from the longest), removes it and checks if a shorter path can be found in the modified graph.
    - The modified graph is constructed by removing edges that are part of "full" SRLGs, i.e., SRLGs whose capacity is fully utilized by the remaining paths.
    - If a shorter path exists, it replaces the removed path; otherwise, the original path is kept.
    - The process repeats until no further shortening is possible.
    Args:
        G (nx.Graph): The input graph, which must have SRLG information in `G.graph["srlgs"]` and edge mapping in `G.graph["edge_by_id"]`.
        s: The source node.
        t: The target node.
        paths (list[list[int]]): A list of paths (each path is a list of node IDs) from `s` to `t`.
    Returns:
        list[list[int]]: The possibly shortened list of paths from `s` to `t`, respecting SRLG constraints.
    """

    # sort the paths by length in descending order
    paths = copy(og_paths)
    paths = sorted(paths, key=lambda path: len(path), reverse=True)

    i = 0
    while i < len(paths):
        # remove the i-th longest path
        new_paths = paths[:i] + paths[i + 1 :]
        deleted_path = paths[i]
        walks_by_edges = make_walks_by_edges(G, new_paths)
        H = G.copy()

        # Remove edges that are "full", i.e., exists a "full" SRLG that contains that edge
        # Full SRLG means, that the number of walks that intersect with it is equal to its capacity
        for srlg in srlgs:

            # Count how many walks this SRLG intersects
            count = sum(
                any(edge in path for edge in srlg["edges"]) for path in walks_by_edges
            )

            # If the SRLG is full, remove its edges from the graph
            if count == srlg["capacity"]:
                for edge in srlg["edges"]:
                    u, v = G.graph["edge_by_id"][edge]
                    if H.has_edge(u, v):
                        H.remove_edge(u, v)

        # Check if there is a path from s to t in the modified graph
        # If there is no path, we cannot shorten the path, so go to the next one
        if not nx.has_path(H, s, t):
            i += 1
            continue

        # If there is a path, find the shortest path from s to t
        new_path = nx.shortest_path(H, s, t)

        # If the new path is not shorter than the deleted path, go to the next one
        if len(new_path) >= len(deleted_path):
            i += 1
            continue

        # If the new path is shorter, replace the deleted path with the new path
        new_paths.append(new_path)
        paths = new_paths
        paths = sorted(paths, key=lambda path: len(path), reverse=True)
        i = 0  # Reset the index to start checking from the beginning again

    return paths


def make_srlgs_tight(
    G: nx.Graph,
    walks: list[list[int]],
    srlgs: list[dict] | None = None,
) -> list[dict]:

    if srlgs is None:
        srlgs = G.graph["srlgs"]

    new_srlgs = deepcopy(srlgs)
    srlgs_intersecting_k_walks = count_srlgs_intersecting_walks(G, walks)
    for k, k_intersections in enumerate(srlgs_intersecting_k_walks):
        for srlg in k_intersections:
            srlg_id = srlg["id"]
            srlg_to_modify = next((x for x in new_srlgs if x["id"] == srlg_id), None)
            assert (
                srlg_to_modify is not None
            ), f"SRLG with id {srlg_id} not found in the list of SRLGs."
            srlg_to_modify["capacity"] = k  # max(k, 1)  # Ensure capacity is at least 1
    return new_srlgs


def improve_surviving_probabilities(G, solution):
    new_solution = deepcopy(solution)
    max_number_of_walks = len(new_solution["walks"][-1])
    for number_of_walks in range(max_number_of_walks, 1, -1):
        # try to improve the surviving probabilities
        if (
            new_solution["surviving_probabilities"][number_of_walks - 1][-1]
            > new_solution["surviving_probabilities"][number_of_walks - 2][-1]
        ):
            for i in range(number_of_walks):
                walks = (
                    new_solution["walks"][number_of_walks - 1][:i]
                    + new_solution["walks"][number_of_walks - 1][i + 1 :]
                )
                sps = get_surviving_probabilities_of_walks(G, walks)
                if (
                    sps[-1]
                    > new_solution["surviving_probabilities"][number_of_walks - 2][-1]
                ):
                    # If we found a better surviving probability, we update the solution
                    new_solution["walks"][number_of_walks - 2] = walks
                    new_solution["surviving_probabilities"][number_of_walks - 2] = sps

    return new_solution


def improve_all_surviving_probabilities(G, solution):
    max_number_of_walks = len(solution["walks"][-1])

    # Make dictionary for improved solution
    # n: number of walks, k: number of survived walks
    improved_solution = {}
    for n in range(1, max_number_of_walks + 1):
        for k in range(1, n + 1):
            improved_solution[(n, k)] = {
                "walks": [],
                "surviving_probability": -1,
                "srlgs": [],
            }

    # Initial solution
    for n in range(1, max_number_of_walks + 1):
        for k in range(1, n + 1):
            improved_solution[(n, k)]["walks"] = solution["walks"][n - 1]
            improved_solution[(n, k)]["surviving_probability"] = solution[
                "surviving_probabilities"
            ][n - 1][k]
            improved_solution[(n, k)]["srlgs"] = solution["srlgs"][n - 1]

    # Improve the solution
    for n in range(max_number_of_walks - 1, 0, -1):
        for k in range(n, 0, -1):
            for i in range(n + 1):  # which walk to remove
                new_walks = (
                    improved_solution[(n + 1, k + 1)]["walks"][:i]
                    + improved_solution[(n + 1, k + 1)]["walks"][i + 1 :]
                )
                new_sps = get_surviving_probabilities_of_walks(G, new_walks)
                # improve diagonal
                if improved_solution[(n, k)]["surviving_probability"] < new_sps[k]:
                    improved_solution[(n, k)]["walks"] = new_walks
                    improved_solution[(n, k)]["surviving_probability"] = new_sps[k]
                    improved_solution[(n, k)]["srlgs"] = improved_solution[
                        (n + 1, k + 1)
                    ]["srlgs"]

                    # improve column
                    for kk in range(k - 1, 0, -1):
                        if (
                            improved_solution[(n, kk)]["surviving_probability"]
                            < new_sps[kk]
                        ):
                            improved_solution[(n, kk)]["walks"] = new_walks
                            improved_solution[(n, kk)]["surviving_probability"] = (
                                new_sps[kk]
                            )
                            improved_solution[(n, kk)]["srlgs"] = improved_solution[
                                (n + 1, k + 1)
                            ]["srlgs"]

    return improved_solution


def improve_all_surviving_probabilities_multiple_methods(G, solutions):
    improved_solutions = {}
    for method in solutions.keys():
        solution = solutions[method]
        max_number_of_walks = len(solution["walks"][-1])

        # Make dictionary for improved solution
        # n: number of walks, k: number of survived walks
        improved_solution = {}
        for n in range(1, max_number_of_walks + 1):
            for k in range(1, n + 1):
                improved_solution[(n, k)] = {
                    "walks": [],
                    "surviving_probability": -1,
                    "srlgs": [],
                }

        # Initial solution
        for n in range(1, max_number_of_walks + 1):
            for k in range(1, n + 1):
                improved_solution[(n, k)]["walks"] = solution["walks"][n - 1]
                improved_solution[(n, k)]["surviving_probability"] = solution[
                    "surviving_probabilities"
                ][n - 1][k]
                improved_solution[(n, k)]["srlgs"] = solution["srlgs"][n - 1]

        # Improve the solution
        for n in range(max_number_of_walks - 1, 0, -1):
            for k in range(n, 0, -1):
                for i in range(n + 1):  # which walk to remove
                    new_walks = (
                        improved_solution[(n + 1, k + 1)]["walks"][:i]
                        + improved_solution[(n + 1, k + 1)]["walks"][i + 1 :]
                    )
                    new_sps = get_surviving_probabilities_of_walks(G, new_walks)
                    # improve diagonal
                    if improved_solution[(n, k)]["surviving_probability"] < new_sps[k]:
                        improved_solution[(n, k)]["walks"] = new_walks
                        improved_solution[(n, k)]["surviving_probability"] = new_sps[k]
                        improved_solution[(n, k)]["srlgs"] = improved_solution[
                            (n + 1, k + 1)
                        ]["srlgs"]

                        # improve column
                        for kk in range(k - 1, 0, -1):
                            if (
                                improved_solution[(n, kk)]["surviving_probability"]
                                < new_sps[kk]
                            ):
                                improved_solution[(n, kk)]["walks"] = new_walks
                                improved_solution[(n, kk)]["surviving_probability"] = (
                                    new_sps[kk]
                                )
                                improved_solution[(n, kk)]["srlgs"] = improved_solution[
                                    (n + 1, k + 1)
                                ]["srlgs"]
        improved_solutions[method] = improved_solution
    return improved_solutions


def dijkstra_heuristic(
    G: nx.Graph, s: int, t: int, srlgs: None | dict = None
) -> list[int]:
    # get tha walk
    H = deepcopy(G)
    if srlgs is None:
        srlgs = G.graph["srlgs"]

    for edge in H.edges():
        intersecting_srlgs = [
            srlg for srlg in srlgs if H.edges[edge]["id"] in srlg["edges"]
        ]
        H.edges[edge]["cost"] = -np.log10(
            1 - sum(srlg["probability"] for srlg in intersecting_srlgs)
        )
    shortest_path = nx.shortest_path(H, s, t, weight="cost")

    # calculate the surviving probability of the path
    surviving_probability = get_surviving_probabilities_of_walks(G, [shortest_path])[-1]

    return shortest_path, surviving_probability


# Plotly functions--------------------------------------------------------------


def get_nodes_trace(
    G: nx.Graph, label: Literal["id", "label", "id+label"] = "label"
) -> go.Scatter:
    """
    Creates a Plotly Scatter trace for the nodes of a NetworkX graph.
    Parameters:
        G (nx.Graph): A NetworkX graph where each node has a "pos" attribute containing its (x, y) coordinates.
    Returns:
        go.Scatter: A Plotly Scatter trace object representing the nodes as markers with optional text.
    """

    node_x = []
    node_y = []
    texts = []
    for node in G.nodes(data=True):
        x, y = node[1]["pos"]
        node_x.append(x)
        node_y.append(y)
        if label == "id":
            texts.append(str(node[0]))
        elif label == "label":
            texts.append(node[1]["label"])
        elif label == "id+label":
            texts.append(f"{node[0]}<br>{node[1]['label']}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=20,
            line_width=1,
            color="lightblue",
        ),
        text=texts,
        textfont=dict(
            color="black",
        ),
        hoverinfo="none",
        name="nodes",
        showlegend=False,
    )
    return node_trace


def get_edges_trace(G: nx.Graph) -> go.Scatter:
    """
    Generates a Plotly Scatter trace representing the edges of a NetworkX graph.
    Args:
        G (nx.Graph): A NetworkX graph where each node has a 'pos' attribute containing its (x, y) coordinates.
    Returns:
        go.Scatter: A Plotly Scatter trace object for visualizing the edges of the graph.
    Notes:
        - The function expects each node in the graph to have a 'pos' attribute as a tuple of (x, y) coordinates.
        - Edges are drawn as straight lines between node positions.
        - The trace is configured to not show hover information and is excluded from the legend.
    """

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]["pos"]
        x1, y1 = G.nodes[edge[1]]["pos"]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)  # append None, so it breaks the line
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888"),
        hoverinfo="none",
        name="edges",
        showlegend=False,
    )

    return edge_trace


def get_edges_markers_trace(G: nx.Graph, resolution: str = RESOLUTION) -> go.Scatter:

    # Get the markers on the edges
    edge_points_x = [
        (
            i * G.nodes[edge[0]]["pos"][0]
            + (resolution + 1 - i) * G.nodes[edge[1]]["pos"][0]
        )
        / (resolution + 1)
        for i in range(1, resolution + 1)
        for edge in G.edges()
    ]
    edge_points_y = [
        (
            i * G.nodes[edge[0]]["pos"][1]
            + (resolution + 1 - i) * G.nodes[edge[1]]["pos"][1]
        )
        / (resolution + 1)
        for i in range(1, resolution + 1)
        for edge in G.edges()
    ]

    edge_marker_trace = go.Scatter(
        x=edge_points_x,
        y=edge_points_y,
        mode="markers",
        marker=dict(size=5, color="lightgrey", opacity=0),
        text=[
            f'Edge id: {data["id"]}'
            for _ in range(resolution)
            for _, _, data in G.edges(data=True)
        ],
        hoverinfo="text",
        name="edge ids on hover",
    )
    return edge_marker_trace


def get_special_path_trace(G: nx.Graph, special_path: list[int]) -> go.Scatter:
    special_path_x = []
    special_path_y = []
    for i in range(len(special_path) - 1):
        x0, y0 = G.nodes[special_path[i]]["pos"]
        x1, y1 = G.nodes[special_path[i + 1]]["pos"]
        special_path_x.append(x0)
        special_path_x.append(x1)
        special_path_y.append(y0)
        special_path_y.append(y1)

    special_path_trace = go.Scatter(
        x=special_path_x,
        y=special_path_y,
        mode="lines",
        line=dict(width=2, color="black", dash="dashdot"),
        hoverinfo="none",
        name="special path",
    )

    return special_path_trace


def get_dual_edge_trace(
    G: nx.PlanarEmbedding, edge: tuple, section: float = 1 / 2
) -> go.Scatter:
    """
    Generate a Plotly Scatter trace representing the dual edge of a given edge in a planar graph.
    This function computes the dual edge trace by traversing the two faces adjacent to the given edge
    in the planar embedding and optionally dividing the edge into sections.
    Args:
        G (nx.PlanarEmbedding): The planar embedding of the graph.
        edge (tuple): A tuple representing the edge (u, v) in the graph.
        section (float, optional): A fraction (0 to 1) indicating the dividing point along the edge.
                                   Defaults to 1/2 (midpoint).
    Returns:
        go.Scatter: A Plotly Scatter object representing the dual edge trace.
    Notes:
        - The function skips the outer face when computing the dual edge trace.
        - The dividing point of the edge is included in the trace.
        - The trace includes `None` values to separate segments for Plotly rendering.
    """
    u, v = edge
    edge_x = []
    edge_y = []
    outer_face = get_outer_face(G)

    # first face
    face_1 = G.traverse_face(u, v)
    if set(face_1) != set(outer_face):
        cx, cy = get_center_of_face(G, face_1)
        edge_x.append(cx)
        edge_y.append(cy)

    # Get the dividing point of the edge
    dx, dy = get_dividing_point_of_edge(G, edge, section)
    edge_x.append(dx)
    edge_y.append(dy)

    # second face
    face_2 = G.traverse_face(v, u)
    if set(face_2) != set(outer_face):
        cx, cy = get_center_of_face(G, face_2)
        edge_x.append(cx)
        edge_y.append(cy)

    # end of the line
    edge_x.append(None)
    edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
    )
    return edge_trace


def get_srlg_traces(
    G: nx.PlanarEmbedding,
    srlgs: list,
    show_all_in_legend: bool = True,
    dash="dash",
    qual_colorscale=qualitative.Plotly,
) -> list[go.Scatter]:
    """
    Generate Plotly Scatter traces for Shared Risk Link Groups (SRLGs) in a graph.
    This function creates a list of Plotly Scatter traces, where each trace represents
    the edges of a specific SRLG in the input graph. The edges are styled and positioned
    based on their SRLG membership and overlap with other SRLGs.
    Args:
        G (nx.PlanarEmbedding): The planar embedding of the graph.
        srlgs (list): A list of SRLGs, where each SRLG is a dictionary containing:
            - "edges": A list of edge IDs that belong to the SRLG.
            - "capacity": The capacity of the SRLG.
            - "color": The color of the SRLG for visualization.
    Returns:
        list[go.Scatter]: A list of Plotly Scatter traces, where each trace corresponds
        to an SRLG. The traces include the x and y coordinates of the edges, their
        styling, and hover information.
    Notes:
        - The function uses `count_intersecting_srlgs` to determine the number of SRLGs
          that share each edge and `get_dual_edge_trace` to compute the trace for each
          edge.
        - Edges that belong to multiple SRLGs are adjusted to avoid overlap in the
          visualization.
    """

    edge_dict = count_intersecting_srlgs(G, srlgs)
    edge_dict = {edge: {"count": edge_dict[edge], "used": 0} for edge in edge_dict}
    srlg_traces = []

    colored_srlgs = deepcopy(srlgs)

    # Assign a unique color to each SRLG using a colorscale if there are more SRLGs than the palette size
    if srlgs and "color" not in srlgs[0]:
        if len(srlgs) <= len(qual_colorscale):
            palette = qual_colorscale
            for i, srlg in enumerate(colored_srlgs):
                srlg["color"] = palette[i % len(palette)]
        else:
            colors = sample_colorscale(
                "Viridis", [i / len(srlgs) for i in range(len(srlgs))]
            )
            for i, srlg in enumerate(colored_srlgs):
                srlg["color"] = colors[i]

    for srlg in colored_srlgs:

        # Create an empty trace for the SRLG
        srlg_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(width=2, color=srlg["color"], dash=dash),
            hoverinfo="text",
            text=f"id: {srlg['id']}",
            name=f"id: {srlg['id']}" if show_all_in_legend else "srlgs",
            legendgroup="srlgs",
            showlegend=show_all_in_legend,
        )

        # Get the endpoints of the edges in the SRLG
        edge_list = [
            (u, v)
            for u, v, data in G.edges(data=True)
            if data["id"] in srlg["edges"] and u > v
        ]

        # Get each edge trace and add it to the SRLG trace
        for edge in edge_list:
            edge_dict[edge]["used"] += 1
            used = edge_dict[edge]["used"]
            count = edge_dict[edge]["count"] + 1

            edge_trace = get_dual_edge_trace(G, edge, used / count)
            srlg_trace["x"] += edge_trace["x"]
            srlg_trace["y"] += edge_trace["y"]

        # Add the SRLG trace to the list of traces
        srlg_traces.append(srlg_trace)

    srlg_traces[0].update(showlegend=True)
    return srlg_traces


def get_disk_srlg_trace(G: nx.Graph, srlgs: list) -> list[go.Scatter]:
    """
    Generates a Plotly Scatter trace and shape definitions for visualizing SRLG (Shared Risk Link Group) disks on a graph.
    Args:
        G (nx.Graph): The networkx graph (not used in the function but kept for interface consistency).
        srlgs (list): A list of SRLG dictionaries, each containing:
            - "center" (tuple): The (x, y) coordinates of the disk center.
            - "radius" (float): The radius of the disk.
            - "probability" (float): The probability associated with the SRLG.
            - "capacity" (optional, float): The capacity value for the SRLG.
            - "color" (str): The fill color for the disk.
    Returns:
        tuple:
            - go.Scatter: A Plotly Scatter trace for the SRLG centers.
            - list: A list of Plotly shape dictionaries representing the SRLG disks as circles.
    """

    xs = []
    ys = []
    texts = []
    shapes = []

    for srlg in srlgs:
        x, y = srlg["center"]
        xs.append(x)
        ys.append(y)
        texts.append(
            f"prob: {srlg['probability']:.2e}<br>cap: {srlg.get('capacity', 0)}"
        )

        shape = dict(
            type="circle",
            x0=round(x - srlg["radius"], 2),
            y0=round(y - srlg["radius"], 2),
            x1=round(x + srlg["radius"], 2),
            y1=round(y + srlg["radius"], 2),
            line=dict(color="lightgrey", width=1),
            fillcolor=srlg["color"],
            opacity=0.2,
            layer="below",
        )
        shapes.append(shape)

    srlg_trace = go.Scatter(
        x=xs,
        y=ys,
        hoverinfo="text",
        text=texts,
        mode="markers",
        marker=dict(size=1, color="grey", opacity=1),
        name="srlgs",
    )

    return srlg_trace, shapes


def get_edge_trace(G, edge, rad=0.1, resolution=RESOLUTION):
    """
    Generate a Plotly Scatter trace representing a curved edge in a graph.
    This function creates a quadratic Bézier curve to represent an edge between two nodes
    in a graph. The curve can be adjusted with a radial offset to avoid overlapping edges.
    Args:
        G (networkx.Graph): The graph containing the nodes and their positions.
            Each node should have a "pos" attribute with its (x, y) coordinates.
        edge (tuple): A tuple (u, v) representing the edge between nodes u and v.
        rad (float, optional): The radial offset for the curve. A positive value
            creates a curve bulging outward, while a negative value creates a curve
            bulging inward. Defaults to 0 (straight line).
        resolution (int): The number of points used to approximate the Bézier curve.
    Returns:
        plotly.graph_objs.Scatter: A Plotly Scatter trace representing the curved edge.
    """

    edge_x = []
    edge_y = []

    u, v = edge
    x0, y0 = G.nodes[u]["pos"]
    x1, y1 = G.nodes[v]["pos"]

    # Midpoint
    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2

    # Perpendicular offset direction
    dx = x1 - x0
    dy = y1 - y0
    length = math.sqrt(dx**2 + dy**2)
    offset_x = -dy / length * rad
    offset_y = dx / length * rad

    # Control point
    ctrl_x = mx + offset_x
    ctrl_y = my + offset_y

    # Get points along the Bézier curve
    x_vals, y_vals = quad_bezier((x0, y0), (ctrl_x, ctrl_y), (x1, y1), resolution)

    # Add the points to the edge trace
    edge_x.extend(x_vals + [None])
    edge_y.extend(y_vals + [None])

    # Create the edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="darkgray"),
        hoverinfo="none",
    )

    return edge_trace


def get_path_traces(G: nx.Graph, paths: list[list[int]]) -> list[go.Scatter]:
    """
    Generate Plotly Scatter traces for paths in a graph.
    Args:
        G (nx.Graph): The planar embedding of the graph.
        paths (list): A list of paths, where each path is represented as a list of node identifiers.
    Returns:
        list[go.Scatter]: A list of Plotly Scatter traces, where each trace corresponds
        to a path. The traces include the x and y coordinates of the nodes, their
        styling, and hover information.
    """
    path_traces = []
    rads = (
        np.linspace(0.1, MAX_RAD, 10)
        if len(paths) <= 10
        else np.linspace(0.1, MAX_RAD, len(paths))
    )
    colors = (
        qualitative.Plotly
        if len(paths) <= 10
        else sample_colorscale("Viridis", [i / len(paths) for i in range(len(paths))])
    )
    for i, path in enumerate(paths):
        path_trace = go.Scatter(
            x=[],
            y=[],
            mode="lines",
            line=dict(width=2, color=colors[i]),
            hoverinfo="none",
            name=f"path #{i}",
            legendgroup="paths",
        )
        for j in range(len(path) - 1):
            u = path[j]
            v = path[j + 1]
            edge_trace = get_edge_trace(G, (u, v), rad=rads[i])
            path_trace["x"] += edge_trace["x"]
            path_trace["y"] += edge_trace["y"]
        path_traces.append(path_trace)
    return path_traces


def get_potential_trace(G: nx.PlanarEmbedding, potentials: list[tuple[int, list[int]]]):
    """
    Generates a Plotly Scatter trace for visualizing potential values at the centers of graph faces, excluding the outer face.
    Args:
        G (nx.PlanarEmbedding): The planar embedding of the graph.
        potentials (list[tuple[int, list[int]]]): A list of tuples, each containing a potential value and a list of node indices representing a face.
    Returns:
        plotly.graph_objs.Scatter: A Plotly Scatter trace with markers and text at the centers of the faces (excluding the outer face), where each marker's text displays the corresponding potential value.
    Notes:
        - The function assumes the existence of `get_outer_face` and `get_center_of_face` helper functions.
        - The outer face is excluded from the visualization.
    """
    # Remove the outer face from the potentials
    outer_face = set(get_outer_face(G))
    potentials = [
        potential for potential in potentials if set(potential[1]) != outer_face
    ]

    node_x = []
    node_y = []
    values = []
    for potential in potentials:
        value = potential[0]
        face = potential[1]
        cx, cy = get_center_of_face(G, face)
        node_x.append(cx)
        node_y.append(cy)
        values.append(value)

    potential_traces = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(size=15, line_width=1, color="plum", symbol="square"),
        text=values,
        hoverinfo="none",
        name="potentials",
    )
    return potential_traces


def create_graph_figure(
    G: nx.Graph, label="label", s=None, t=None, title=None
) -> go.Figure:
    # Get traces
    edge_trace = get_edges_trace(G)
    node_trace = get_nodes_trace(G, label=label)
    traces = [edge_trace, node_trace]

    # Gather info
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_srlgs = len(G.graph.get("srlgs", []))

    title_text = title if title else f'{G.graph.get("name", "Graph")}'

    # Create figure
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text=title_text, font=dict(size=16)),
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"
            ),
            annotations=[
                dict(
                    text=f"Nodes: {num_nodes}<br>Edges: {num_edges}<br>SRLGs: {num_srlgs}",
                    xref="paper",
                    yref="paper",
                    x=1,
                    y=0,
                    xanchor="right",
                    yanchor="bottom",
                    showarrow=False,
                    font=dict(size=12),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                )
            ],
        ),
    )

    if s is not None and t is not None:  # Change the colors of node s and t
        node_colors = ["lightblue"] * len(node_trace.x)
        s_index = list(node_trace.text).index(str(s))
        t_index = list(node_trace.text).index(str(t))
        node_colors[s_index] = "lightpink"
        node_colors[t_index] = "lightpink"
        fig.data[1].marker.color = node_colors
        fig.update_layout(
            legend=dict(
                x=0.2,
                y=0.99,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=12),
            )
        )

    return fig


def create_paths_figure(G, special_path, potentials, walks, title):
    # Create a planar embedding of the graph
    H = planar_embed(G)

    # Get traces
    edge_trace = get_edges_trace(G)
    node_trace = get_nodes_trace(G, label="label")
    special_path_trace = get_special_path_trace(G, special_path)
    potential_trace = get_potential_trace(H, potentials)
    # Group walks in legend
    path_traces = get_path_traces(G, walks)
    for path in path_traces[1:]:
        path.showlegend = False
    path_traces[0].name = "paths"
    # Combine all traces
    traces = path_traces + [edge_trace, special_path_trace, potential_trace, node_trace]

    # Make the figure
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text=title),
            margin=dict(b=0, l=0, r=0, t=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"
            ),
        ),
    )

    # Make the figure a standalone object
    return fig


def save_figures_to_html(figures: list[go.Figure], filename: str, title: str):
    # Save all figures in one HTML file (2 columns via simple CSS)
    file_path = os.path.join("figures", "htmls", filename)
    with open(file_path, "w") as f:
        f.write(
            f"""
        <html>
        <head>
            <title>{{ {title} }}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                .container {{ display: flex; flex-wrap: wrap; }}
                .plot-box {{ width: 48%; padding: 1%; box-sizing: border-box; }}
            </style>
        </head>
        <body>
            <div class="container">
        """
        )

        for fig in figures:
            inner_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)
            f.write(f'<div class="plot-box">{inner_html}</div>')

        f.write(
            """
            </div>
        </body>
        </html>
        """
        )


def create_terminal_distance_histogram(G: nx.Graph) -> go.Figure:
    """
    Creates a histogram of terminal-to-terminal shortest path distances in the graph.
    Args:
        G (nx.Graph): The input graph.
    Returns:
        go.Figure: A Plotly figure object containing the histogram.
    """
    # Get all pairs of nodes (excluding self-pairs)
    node_pairs = [(u, v) for u in G.nodes for v in G.nodes if u != v]

    # Compute shortest path lengths for all pairs
    distances = []
    for u, v in node_pairs:
        dist = nx.shortest_path_length(G, source=u, target=v)
        distances.append(dist)

    # Count occurrences of each distance
    distance_counts = Counter(distances)

    # Prepare data for plotting
    x = sorted(distance_counts.keys())
    y = [distance_counts[d] // 2 for d in x]

    fig = go.Figure([go.Bar(x=x, y=y)])
    fig.update_layout(
        title="Number of (s, t) pairs by shortest path distance",
        xaxis_title="Shortest Path Distance",
        yaxis_title="Number of (s, t) Pairs",
        margin=dict(b=60, t=60, l=60, r=60),  # add some margin
        xaxis=dict(
            tickmode="linear", tick0=min(x), dtick=1, range=[min(x) - 0.5, max(x) + 0.5]
        ),  # extend x axis range
    )
    fig.update_traces(
        text=y, textposition="outside", cliponaxis=False
    )  # allow text outside plot area
    return fig


# Dash functions----------------------------------------------------------------


def make_figure_with_shapes(traces, shapes, show_shapes=True):
    return go.Figure(
        data=traces if show_shapes else traces[1:],
        layout=go.Layout(
            title=dict(text="Test graph", font=dict(size=16)),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"
            ),
            shapes=shapes if show_shapes else [],
        ),
    )


def make_figure(traces):
    return go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(text="Test graph", font=dict(size=16)),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=0, l=0, r=0, t=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x"
            ),
        ),
    )


# IO functions------------------------------------------------------------------


def read_italy_from_gml(path: str) -> nx.graph:
    G = nx.read_gml(path, label="id")
    G.graph["name"] = G.graph.pop("label")

    # Create a mapping of edge IDs to the corresponding nodes
    G.graph["edge_by_id"] = {data["id"]: (u, v) for u, v, data in G.edges(data=True)}
    for _, data in G.nodes(data=True):
        data.pop("weight", None)

    for node, data in G.nodes(data=True):
        # Assign a position to each node if not already present
        lat = data.pop("Latitude")
        lon = data.pop("Longitude")
        data["pos"] = (lon, lat)

    return G


def read_graph_from_gml(path: str) -> nx.Graph:
    """
    Reads a graph from a GML file and assigns unique IDs to each edge.
    Args:
        path (str): The file path to the GML file.
    Returns:
        G (nx.Graph): A NetworkX graph object with edges having unique IDs.
    """

    # Read gml file
    G = nx.read_gml(path, label="id")
    G.graph["name"] = G.graph.pop("label")
    G.graph["center"] = float(G.graph.pop("center_lon")), float(
        G.graph.pop("center_lat")
    )
    G.graph["rate"] = float(G.graph.pop("annual_rate"))

    # Assign unique IDs to edges
    for edge_id, (_, _, data) in enumerate(G.edges(data=True)):
        # Assign a unique ID to the edge
        data["id"] = edge_id

    # Create a mapping of edge IDs to the corresponding nodes
    G.graph["edge_by_id"] = {data["id"]: (u, v) for u, v, data in G.edges(data=True)}

    for node, data in G.nodes(data=True):
        # Assign a position to each node if not already present
        lat = data.pop("Latitude")
        lon = data.pop("Longitude")
        data["pos"] = (lon, lat)

    return G


def read_srlgs_from_xml(path: str, G: nx.graph) -> list[dict]:
    """
    Reads a PSRLG (Probabilistic Shared Risk Link Group) XML file and processes the data into a list of dictionaries.
    Args:
        path (str): The file path to the PSRLG XML file.
        G (nx.Graph): A NetworkX graph object containing the network topology.
    Returns:
        psrlg_data (List[Dict[str, Union[List[int], float]]]): A list of dictionaries, each containing:
            - "edges" (List[int]): A list of edge IDs that belong to the PSRLG.
            - "probability" (float): The probability associated with the PSRLG.
    Notes:
        - The function assumes that the XML file contains PSRLG groups with "Edges" and "Probability" elements.
        - The "Edges" element should contain edge descriptions in the format "id:(node1, node2)".
        - The function retrieves the edge ID from the graph `G` using the node pair (node1, node2).
        - The resulting list of PSRLG data is sorted in descending order of probability.
    """

    # read XML file
    tree = ET.parse(path)
    root = tree.getroot()

    # processing SRLG groups
    srlgs = []
    id = 0
    for psrlg in root.findall(".//PSRLG"):
        edges_text = psrlg.find("Edges").text.strip()
        probability = float(psrlg.find("Probability").text.strip())

        # Processing edges (e.g. "9:(18, 17)" -> id of (18, 17))
        # WARNING: edge ids from the XML file are not matching the ids in the graph
        edges = []
        for line in edges_text.split("\n"):
            parts = line.split(":")
            if len(parts) == 2:
                nodes = tuple(map(int, parts[1].strip("()").split(",")))
                edge_id = G.get_edge_data(*nodes)["id"]  # Él lekérdezése
                edges.append(edge_id)

        # append the SRLG to the list if it has edges
        if edges:
            srlgs.append({"id": id, "edges": edges, "probability": probability})
            id += 1

    return srlgs


def read_disk_srlgs_from_json(path: str) -> list[dict]:
    """
    Reads SRLG (Shared Risk Link Group) data from a JSON file, skipping the first line which contains metadata.
    Args:
        path (str): The file path to the JSON file containing SRLG data.
    Returns:
        list[dict]: A list of dictionaries representing the SRLGs read from the file.
    """

    with open(path, "r") as file:
        next(file)  # skip the first line, its metadata about the generation
        srlgs = json.load(file)
    return srlgs


# C++ communication functions---------------------------------------------------


def generate_input_string(
    G: nx.Graph, source_node: int, target_node: int, srlgs: list[dict] | None = None
) -> str:
    """
    Generates an input string for the C++ code
    Args:
        G (nx.Graph): The input graph.
    Returns:
        str: The generated input string.
    """
    input_string = ""
    # Nodes
    input_string += f"{len(G.nodes())}\n"
    # Nodes in format: x y id
    for node in G.nodes(data=True):
        input_string += f"{node[1]['pos'][0]} {node[1]['pos'][1]} {node[0]}\n"
    input_string += f"{source_node} {target_node}\n"

    # Edges
    input_string += f"{len(G.edges())}\n"
    # Edges in format: u v id
    for edge in G.edges(data=True):
        input_string += f"{edge[0]} {edge[1]} {edge[2]['id']}\n"

    # SRLGS
    if srlgs is None:
        input_string += f"{len(G.graph['srlgs'])}\n"
        # SRLGs in format: length edge_ids capacity
        for srlg in G.graph["srlgs"]:
            input_string += f"{len(srlg['edges'])} {' '.join(map(str, srlg['edges']))} {srlg['capacity']}\n"
    else:
        input_string += f"{len(srlgs)}\n"
        for srlg in srlgs:
            input_string += f"{len(srlg['edges'])} {' '.join(map(str, srlg['edges']))} {srlg['capacity']}\n"
    return input_string


def generate_input_string_with_prob(
    G: nx.Graph, source_node: int, target_node: int, srlgs: list[dict] | None = None
) -> str:
    """
    Generates an input string for the C++ code
    Args:
        G (nx.Graph): The input graph.
    Returns:
        str: The generated input string.
    """
    input_string = ""
    # Nodes
    input_string += f"{len(G.nodes())}\n"
    # Nodes in format: x y id
    for node in G.nodes(data=True):
        input_string += f"{node[1]['pos'][0]} {node[1]['pos'][1]} {node[0]}\n"
    input_string += f"{source_node} {target_node}\n"

    # Edges
    input_string += f"{len(G.edges())}\n"
    # Edges in format: u v id
    for edge in G.edges(data=True):
        input_string += f"{edge[0]} {edge[1]} {edge[2]['id']}\n"

    # SRLGS
    if srlgs is None:
        input_string += f"{len(G.graph['srlgs'])}\n"
        # SRLGs in format: length edge_ids capacity
        for srlg in G.graph["srlgs"]:
            input_string += f"{len(srlg['edges'])} {' '.join(map(str, srlg['edges']))} {srlg['capacity']} {srlg['probability']} {srlg['id']}\n"
    else:
        input_string += f"{len(srlgs)}\n"
        for srlg in srlgs:
            input_string += f"{len(srlg['edges'])} {' '.join(map(str, srlg['edges']))} {srlg['capacity']} {srlg['probability']} {srlg['id']}\n"
    return input_string


def solve_from_string(exe_path: str, input_string: str, print_output: bool = False):
    """
    Executes an external program with the given input string and parses its output into structured data.
    Args:
        exe_path (str): The path to the executable to run.
        input_string (str): The input string to pass to the executable's stdin.
        print_output (bool): If True, prints the output of the executable to stdout.
    Returns:
        tuple:
            special_path (list of int): The special path parsed from the output.
            paths (list of list of int): A list of paths, each represented as a list of integers.
            potentials (list of tuple): A list of tuples, each containing an integer and a list of integers, representing potentials.
            cycle (list of list of int): The negative cycle, represented as a list of lists of integers.
    Raises:
        ValueError: If the output format is not as expected or cannot be parsed.
    """

    result = subprocess.run(
        [exe_path], input=input_string, text=True, capture_output=True
    )
    if print_output:
        print("Return code:", result.returncode)
        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)
    try:
        all_paths_string, temp = result.stdout.split("potentials:\n")
        potentials_string, negative_cycle_string = temp.split("negative cycle:\n")

        # handle remainder
        all_paths = all_paths_string.splitlines()[1:]

        # get special path
        special_path = list(map(int, all_paths[0].split()))

        # get the paths
        paths = [list(map(int, line.split())) for line in all_paths[2:]]

        # get the potentials
        potentials = [
            (int(parts[0]), list(map(int, parts[1:])))
            for parts in (
                line.split() for line in potentials_string.strip().splitlines()
            )
        ]

        # get the negative cycle
        cycle = [
            list(map(int, line.split()))
            for line in negative_cycle_string.split("\n")[0:-1]
        ]

        return special_path, paths, potentials, cycle

    except:
        if not print_output:
            print("Return code:", result.returncode)
            print("Standard Output:\n", result.stdout)
            print("Standard Error:\n", result.stderr)


def solve_from_string_prob(
    exe_path: str,
    input_string: str,
    method: str = "min_cutting_prob",
    number_of_backups: str = 0,
    max_number_of_paths: int = 10,
    print_output: bool = False,
    edge_disjoint=False,
):
    """
    Executes an external program with the given input string and parses its output into structured data.
    Args:
        exe_path (str): The path to the executable to run.
        input_string (str): The input string to pass to the executable's stdin.
        print_output (bool): If True, prints the output of the executable to stdout.
    Returns:
        tuple:
            special_path (list of int): The special path parsed from the output.
            paths (list of list of int): A list of paths, each represented as a list of integers.
            potentials (list of tuple): A list of tuples, each containing an integer and a list of integers, representing potentials.
            cycle (list of list of int): The negative cycle, represented as a list of lists of integers.
    Raises:
        ValueError: If the output format is not as expected or cannot be parsed.
    """
    if edge_disjoint:
        result = subprocess.run(
            [
                exe_path,
                method,
                str(number_of_backups),
                str(max_number_of_paths),
                "edge_disjoint",
            ],
            input=input_string,
            text=True,
            capture_output=True,
        )
    else:
        result = subprocess.run(
            [exe_path, method, str(number_of_backups), str(max_number_of_paths)],
            input=input_string,
            text=True,
            capture_output=True,
        )
    if print_output:
        print("Return code:", result.returncode)
        print("Standard Output:\n", result.stdout)
        print("Standard Error:\n", result.stderr)
    try:
        solution = {}
        lines = result.stdout.splitlines()
        is_path = True
        # Skip the first line which is just "first st path:"
        special_path = list(map(int, lines[1].split()))
        for line in lines[2:]:
            if line.startswith("number of paths: "):
                number_of_paths = int(line.split()[-1])
                solution[number_of_paths] = {"paths": [], "srlgs": {}}
                is_path = True
            elif line.startswith("SRLG capacities: "):
                is_path = False
            else:
                if is_path:
                    path = list(map(int, line.split()))
                    solution[number_of_paths]["paths"].append(path)
                else:
                    srlg_id, capacity = map(int, line.split())
                    solution[number_of_paths]["srlgs"][srlg_id] = capacity

        return solution

    except:
        if not print_output:
            print("Return code:", result.returncode)
            print("Standard Output:\n", result.stdout)
            print("Standard Error:\n", result.stderr)


# Archive functions (currently not used)----------------------------------------


def make_walks(G: nx.graph, walks_by_edges: list):
    # TODO not working when the path is only one edge
    # create the inverse list of edges
    walks_by_nodes = []
    for edges in walks_by_edges:
        prev_edges = [-1] * len(edges)
        for i, next_edge in enumerate(edges):
            if next_edge != -1:  # -1 means no next edge
                prev_edges[next_edge] = i

        # Create a walk from source_node to target_node using the edges list.
        walk = []
        circles = []
        for current_edge in range(len(edges)):

            if edges[current_edge] != -1:
                entry_edge = current_edge
                next_edge = edges[current_edge]

                # delete this edge from the list
                edges[current_edge] = -1
                prev_edges[next_edge] = -1

                new_walk = deque()
                # get the endpoints of the current edge
                u, v = G.graph["edge_by_id"][current_edge]

                x, y = G.graph["edge_by_id"][next_edge]
                # append starting node to the new walk
                if u == x or u == y:
                    new_walk.append(v)
                    new_walk.append(u)
                else:
                    new_walk.append(u)
                    new_walk.append(v)
                # edges forward
                while next_edge != -1:
                    current_edge = next_edge
                    next_edge = edges[current_edge]

                    edges[current_edge] = -1
                    prev_edges[next_edge] = -1

                    if current_edge == entry_edge:
                        break

                    # append the new endpoint to the walk
                    x, y = G.graph["edge_by_id"][current_edge]
                    if x == new_walk[-1]:
                        new_walk.append(y)
                    else:
                        new_walk.append(x)

                # edges backward
                current_edge = entry_edge
                prev_edge = prev_edges[current_edge]
                while prev_edge != -1:
                    # remove the edge from the list
                    edges[prev_edge] = -1
                    prev_edges[current_edge] = -1

                    # go backward
                    current_edge = prev_edge
                    prev_edge = prev_edges[current_edge]

                    # append the new endpoint to the walk
                    x, y = G.graph["edge_by_id"][current_edge]
                    if x == new_walk[0]:
                        new_walk.appendleft(y)
                    else:
                        new_walk.appendleft(x)

                # if this is the start of the walk
                if new_walk[0] != new_walk[-1]:
                    walk = list(new_walk)
                else:
                    circles.append(list(new_walk)[:-1])

        for circle in circles:
            intersection = list(set(walk) & set(circle))[0]
            walk_idx = walk.index(intersection)
            circle_idx = circle.index(intersection)
            # Rotate the circle so that circle[circle_idx] is at the start
            rotated_circle = (
                circle[circle_idx:] + circle[:circle_idx] + [circle[circle_idx]]
            )
            walk = walk[:walk_idx] + rotated_circle + walk[walk_idx + 1 :]
        walks_by_nodes.append(walk)
    return walks_by_nodes
