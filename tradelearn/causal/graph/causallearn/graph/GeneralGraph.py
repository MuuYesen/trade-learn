#!/usr/bin/env python3
from __future__ import annotations

from abc import ABC
from typing import List, Dict, Tuple

import numpy as np
from numpy import ndarray
from .Edge import Edge
from .Endpoint import Endpoint
from .Graph import Graph
from .Node import Node
from tradelearn.causal.graph.causallearn.utils.GraphUtils import GraphUtils


# Represents a graph using a matrix. Variables are permitted to be either measured
# or latent, with multiple edges allowed per node pair (to allow for two-cycles and
# unmeasured confounders) and no edges to self. Allowable edge types:
# ---
# -->
# <->
# --o
# o-o
class GeneralGraph(Graph, ABC):

    def __init__(self, nodes: List[Node]):
        self.nodes: List[Node] = nodes
        self.num_vars: int = len(nodes)

        node_map: Dict[Node, int] = {}

        for i in range(self.num_vars):
            node = nodes[i]
            node_map[node] = i

        self.node_map: Dict[Node, int] = node_map

        self.graph: ndarray = np.zeros((self.num_vars, self.num_vars), np.dtype(int))
        self.dpath: ndarray = np.zeros((self.num_vars, self.num_vars), np.dtype(int))

        self.reconstitute_dpath([])

        self.ambiguous_triples: List[Tuple[Node, Node, Node]] = []
        self.underline_triples: List[Tuple[Node, Node, Node]] = []
        self.dotted_underline_triples: List[Tuple[Node, Node, Node]] = []

        self.attributes = {}
        self.pattern = False
        self.pag = False

    ### Helper Functions ###

    def adjust_dpath(self, i: int, j: int):
        dpath = self.dpath
        dpath[j, i] = 1

        for k in range(self.num_vars):
            if dpath[i, k] == 1:
                dpath[j, k] = 1

            if dpath[k, j] == 1:
                dpath[k, i] = 1

        self.dpath = dpath

    def reconstitute_dpath(self, edges: List[Edge]):
        self.dpath = np.zeros((self.num_vars, self.num_vars), np.dtype(int))
        for i in range(self.num_vars):
            self.adjust_dpath(i, i)

        while len(edges) > 0:
            edge = edges.pop()
            node1 = edge.get_node1()
            node2 = edge.get_node2()
            i = self.node_map[node1]
            j = self.node_map[node2]
            if self.is_parent_of(node1, node2):
                self.adjust_dpath(i, j)
            elif self.is_parent_of(node2, node1):
                self.adjust_dpath(j, i)


    def collect_ancestors(self, node: Node, ancestors: List[Node]):
        if node in ancestors:
            return

        ancestors.append(node)
        parents = self.get_parents(node)

        if parents:
            for parent in parents:
                self.collect_ancestors(parent, ancestors)

    ### Public Functions ###

    # Adds a directed edge --> to the graph.
    def add_directed_edge(self, node1: Node, node2: Node):
        i = self.node_map[node1]
        j = self.node_map[node2]
        self.graph[j, i] = 1
        self.graph[i, j] = -1

        self.adjust_dpath(i, j)

    # Adds the specified edge to the graph, provided it is not already in the
    # graph.
    def add_edge(self, edge: Edge):
        node1 = edge.get_node1()
        node2 = edge.get_node2()
        endpoint1 = str(edge.get_endpoint1())
        endpoint2 = str(edge.get_endpoint2())

        i = self.node_map[node1]
        j = self.node_map[node2]

        e1 = self.graph[i, j]
        e2 = self.graph[j, i]

        bidirected = e2 == 1 and e1 == 1
        existing_edge = not bidirected and (e2 != 0 or e1 != 0)

        if endpoint1 == "TAIL":
            if existing_edge:
                return False
            if endpoint2 == "TAIL":
                if bidirected:
                    self.graph[j, i] = Endpoint.TAIL_AND_ARROW.value
                    self.graph[i, j] = Endpoint.TAIL_AND_ARROW.value
                else:
                    self.graph[j, i] = -1
                    self.graph[i, j] = -1
            else:
                if endpoint2 == "ARROW":
                    if bidirected:
                        self.graph[j, i] = Endpoint.ARROW_AND_ARROW.value
                        self.graph[i, j] = Endpoint.TAIL_AND_ARROW.value
                    else:
                        self.graph[j, i] = 1
                        self.graph[i, j] = -1
                    self.adjust_dpath(i, j)
                else:
                    if endpoint2 == "CIRCLE":
                        if bidirected:
                            return False
                        else:
                            self.graph[j, i] = 2
                            self.graph[i, j] = -1
                    else:
                        return False
        else:
            if endpoint1 == "ARROW":
                if endpoint2 == "ARROW":
                    if existing_edge:

                        if e1 == 2 or e2 == 2:
                            return False
                        if self.graph[j, i] == Endpoint.ARROW.value:
                            self.graph[j, i] = Endpoint.ARROW_AND_ARROW.value
                        else:
                            self.graph[j, i] = Endpoint.TAIL_AND_ARROW.value
                        if self.graph[i, j] == Endpoint.ARROW.value:
                            self.graph[i, j] = Endpoint.ARROW_AND_ARROW.value
                        else:
                            self.graph[i, j] = Endpoint.TAIL_AND_ARROW.value
                    else:
                        self.graph[j, i] = Endpoint.ARROW.value
                        self.graph[i, j] = Endpoint.ARROW.value
                else:
                    return False
            else:
                if endpoint1 == "CIRCLE":
                    if existing_edge:
                        return False
                    if endpoint2 == "ARROW":
                        if bidirected:
                            return False
                        else:
                            self.graph[j, i] = 1
                            self.graph[i, j] = 2
                    else:
                        if endpoint2 == "CIRCLE":
                            if bidirected:
                                return False
                            else:
                                self.graph[j, i] = 2
                                self.graph[i, j] = 2
                        else:
                            return False
                else:
                    return False

            return True

    # Adds a node to the graph. Precondition: The proposed name of the node
    # cannot already be used by any other node in the same graph.
    def add_node(self, node: Node) -> bool:
        if node in self.nodes:
            return False

        nodes = self.nodes
        nodes.append(node)
        self.nodes = nodes

        self.num_vars = self.num_vars + 1

        self.node_map[node] = self.num_vars - 1

        row = np.zeros(self.num_vars - 1)
        graph = np.vstack((self.graph, row))
        dpath = np.vstack((self.dpath, row))

        col = np.zeros(self.num_vars)
        graph = np.column_stack((graph, col))
        dpath = np.column_stack((dpath, col))

        self.graph = graph
        self.dpath = dpath

        self.adjust_dpath(self.num_vars - 1, self.num_vars - 1)

        return True

    # Removes all nodes (and therefore all edges) from the graph.
    def clear(self):
        self.nodes = []
        self.num_vars = 0
        self.node_map = {}
        self.graph = np.zeros((self.num_vars, self.num_vars), np.dtype(int))
        self.dpath = np.zeros((self.num_vars, self.num_vars), np.dtype(int))

    # Determines whether this graph contains the given edge.
    #
    # Returns true iff the graph contain 'edge'.
    def contains_edge(self, edge: Edge) -> bool:
        endpoint1 = str(edge.get_endpoint1())
        endpoint2 = str(edge.get_endpoint2())

        node1 = edge.get_node1()
        node2 = edge.get_node2()

        i = self.node_map[node1]
        j = self.node_map[node2]

        e1 = self.graph[i, j]
        e2 = self.graph[j, i]

        if endpoint1 == "TAIL":
            if endpoint2 == "TAIL":
                if (e2 == -1 and e1 == -1) \
                        or (e2 == Endpoint.TAIL_AND_ARROW.value and e1 == Endpoint.TAIL_AND_ARROW.value):
                    return True
                else:
                    return False
            else:
                if endpoint2 == "ARROW":
                    if (e1 == -1 and e2 == 1) \
                            or (e1 == Endpoint.TAIL_AND_ARROW.value and e2 == Endpoint.ARROW_AND_ARROW.value):
                        return True
                    else:
                        return False
                else:
                    if endpoint2 == "CIRCLE":
                        if e1 == -1 and e2 == 2:
                            return True
                        else:
                            return False
                    else:
                        return False
        else:
            if endpoint1 == "ARROW":
                if endpoint2 == "ARROW":
                    if (e1 == Endpoint.ARROW.value and e2 == Endpoint.ARROW.value) \
                            or (e1 == Endpoint.TAIL_AND_ARROW.value and e2 == Endpoint.TAIL_AND_ARROW.value) \
                            or (e1 == Endpoint.ARROW_AND_ARROW.value or e2 == Endpoint.ARROW_AND_ARROW.value):
                        return True
                    else:
                        return False
                else:
                    return False
            else:
                if endpoint1 == "CIRCLE":
                    if endpoint2 == "ARROW":
                        if e1 == 2 and e2 == 1:
                            return True
                        else:
                            return False
                    else:
                        if endpoint2 == "CIRCLE":
                            if e1 == 2 and e2 == 2:
                                return True
                            else:
                                return False
                        else:
                            return False
                else:
                    return False

    # Determines whether this graph contains the given node.
    #
    # Returns true iff the graph contains 'node'.
    def contains_node(self, node: Node) -> bool:
        node_list = self.nodes
        return node in node_list

    # Returns true iff there is a directed cycle in the graph.
    def exists_directed_cycle(self) -> bool:
        utils = GraphUtils()
        for node in self.nodes:
            if utils.exists_directed_path_from_to_breadth_first(node, node, self):
                return True

        return False

    # Returns true iff a trek exists between two nodes in the graph.  A trek
    # exists if there is a directed path between the two nodes or else, for
    # some third node in the graph, there is a path to each of the two nodes in
    # question.
    def exists_trek(self, node1: Node, node2: Node) -> bool:
        for node in self.nodes:
            if self.is_ancestor_of(node, node1) and self.is_ancestor_of(node, node2):
                return True

        return False

    # Determines whether this graph is equal to some other graph, in the sense
    # that they contain the same nodes and the sets of edges defined over these
    # nodes in the two graphs are isomorphic typewise. That is, if node A and B
    # exist in both graphs, and if there are, e.g., three edges between A and B
    # in the first graph, two of which are directed edges and one of which is
    # an undirected edge, then in the second graph there must also be two
    # directed edges and one undirected edge between nodes A and B.
    def __eq__(self, other):
        if isinstance(other, GeneralGraph):
            sorted_list = self.nodes.sort()
            if sorted_list == other.nodes.sort() and np.array_equal(self.graph, other.graph):
                return True
            else:
                return False
        else:
            return False

    # Returns a mutable list of nodes adjacent to the given node.
    def get_adjacent_nodes(self, node: Node) -> List[Node]:
        j = self.node_map[node]
        adj_list: List[Node] = []

        for i in range(self.num_vars):
            if (not self.graph[j, i] == 0) and (not self.graph[i, j] == 0):
                node2 = self.nodes[i]
                adj_list.append(node2)

        return adj_list

    # Return the list of parents of a node.
    def get_parents(self, node) -> List[Node]:
        j = self.node_map[node]
        parents: List[Node] = []

        for i in range(self.num_vars):
            if (self.graph[i, j] == -1 and self.graph[j, i] == 1) \
                    or (self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value
                        and self.graph[j, i] == Endpoint.ARROW_AND_ARROW.value):
                node2 = self.nodes[i]
                parents.append(node2)

        return parents

    # Returns a mutable list of ancestors for the given nodes.
    def get_ancestors(self, nodes: List[Node]) -> List[Node]:
        if not isinstance(nodes, list):
            raise TypeError("Must be a list of nodes")

        ancestors: List[Node] = []

        for node in nodes:
            self.collect_ancestors(node, ancestors)

        return ancestors

    # Returns a mutable list of children for a node.
    def get_children(self, node: Node) -> List[Node]:
        i = self.node_map[node]
        children: List[Node] = []

        for j in range(self.num_vars):
            if (self.graph[j, i] == 1 and self.graph[i, j] == -1) \
                    or (self.graph[j, i] == Endpoint.ARROW_AND_ARROW.value
                        and self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value):
                node2 = self.nodes[j]
                children.append(node2)

        return children

    # Returns the number of arrow endpoints adjacent to the node.
    def get_indegree(self, node: Node) -> int:
        i = self.node_map[node]
        indegree = 0

        for j in range(self.num_vars):
            if self.graph[i, j] == 1:
                indegree = indegree + 1
            else:
                if self.graph[i, j] == Endpoint.ARROW_AND_ARROW.value:
                    indegree = indegree + 2

        return indegree

    # Returns the number of null endpoints adjacent to the node.
    def get_outdegree(self, node: Node) -> int:
        i = self.node_map[node]
        outdegree = 0

        for j in range(self.num_vars):
            if self.graph[i, j] == -1 or self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value:
                outdegree = outdegree + 1

        return outdegree

    # Returns the total number of edges into and out of the node.
    def get_degree(self, node: Node) -> int:
        i = self.node_map[node]
        degree = 0

        for j in range(self.num_vars):
            if self.graph[i, j] == 1 or self.graph[i, j] == -1 or self.graph[i, j] == 2:
                degree = degree + 1
            else:
                if self.graph[i, j] != 0:
                    degree = degree + 2

        return degree

    def get_max_degree(self) -> int:
        nodes = self.nodes
        max_degree = -1

        for node in nodes:
            deg = self.get_degree(node)
            if deg > max_degree:
                max_degree = deg

        return max_degree

    # Returns the node with the given string name.  In case of accidental
    # duplicates, the first node encountered with the given name is returned.
    # In case no node exists with the given name, None is returned.
    def get_node(self, name: str) -> Node | None:
        for node in self.nodes:
            if node.get_name() == name:
                return node

        return None

    # Returns the list of nodes for the graph.
    def get_nodes(self) -> List[Node]:
        return self.nodes

    # Returns the names of the nodes, in the order of get_nodes.
    def get_node_names(self) -> List[str]:
        node_names: List[str] = []
        for node in self.nodes:
            node_names.append(node.get_name())
        return node_names

    # Returns the number of edges in the entire graph.
    def get_num_edges(self) -> int:
        edges = 0
        for i in range(self.num_vars):
            for j in range(i + 1, self.num_vars):
                if self.graph[i, j] == 1 or self.graph[i, j] == -1 or self.graph[i, j] == 2:
                    edges = edges + 1
                else:
                    if self.graph[i, j] != 0:
                        edges = edges + 2

        return edges

    # Returns the number of edges in the graph which are connected to a particular node.
    def get_num_connected_edges(self, node: Node) -> int:
        i = self.node_map[node]
        edges = 0
        for j in range(self.num_vars):
            if self.graph[j, i] == 1 or self.graph[j, i] == -1 or self.graph[j, i] == 2:
                edges = edges + 1
            else:
                if self.graph[j, i] != 0:
                    edges = edges + 2

        return edges

    # Return the number of nodes in the graph.
    def get_num_nodes(self) -> int:
        return self.num_vars

    # Return true iff node1 is adjacent to node2 in the graph.
    def is_adjacent_to(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]
        return self.graph[j, i] != 0

    # Return true iff node1 is an ancestor of node2.
    def is_ancestor_of(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]
        return self.dpath[j, i] == 1

    # Return true iff node1 is a child of node2.
    def is_child_of(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]
        return (self.graph[j, i] == Endpoint.TAIL.value and self.graph[i, j] == Endpoint.ARROW.value) \
               or self.graph[j, i] == Endpoint.TAIL_AND_ARROW.value

    # Returns true iff node1 is a parent of node2.
    def is_parent_of(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]
        return (self.graph[j, i] == Endpoint.ARROW.value and self.graph[i, j] == Endpoint.TAIL.value) \
               or self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value

    # Returns true iff node1 is a proper ancestor of node2.
    def is_proper_ancestor_of(self, node1: Node, node2: Node) -> bool:
        return self.is_ancestor_of(node1, node2) and not (node1 == node2)

    # Returns true iff node1 is a proper descendant of node2.
    def is_proper_descendant_of(self, node1: Node, node2: Node) -> bool:
        return self.is_descendant_of(node1, node2) and not (node1 == node2)

    # Returns true iff node1 is a descendant of node2.
    def is_descendant_of(self, node1: Node, node2: Node) -> bool:
        return self.is_ancestor_of(node2, node1)

    # Returns the edge connecting node1 and node2, provided a unique such edge exists.
    def get_edge(self, node1: Node, node2: Node) -> Edge | None:
        i = self.node_map[node1]
        j = self.node_map[node2]

        end_1 = self.graph[i, j]
        end_2 = self.graph[j, i]

        if end_1 == 0:
            return None

        edge = Edge(node1, node2, Endpoint(end_1), Endpoint(end_2))
        return edge

    # Returns the directed edge from node1 to node2, if there is one.
    def get_directed_edge(self, node1: Node, node2: Node) -> Edge | None:
        i = self.node_map[node1]
        j = self.node_map[node2]

        end_1 = self.graph[i, j]
        end_2 = self.graph[j, i]

        if end_1 > 1 or end_1 == 0 or (end_1 == -1 and end_2 == -1):
            return None

        edge = Edge(node1, node2, Endpoint(end_1), Endpoint(end_2))
        return edge

    # Returns the list of edges connected to a particular node.
    # No particular ordering of the edges in the list is guaranteed.
    def get_node_edges(self, node: Node) -> List[Edge]:
        i = self.node_map[node]
        edges: List[Edge] = []

        for j in range(self.num_vars):
            node2 = self.nodes[j]
            if self.graph[j, i] == 1 or self.graph[j, i] == -1 or self.graph[j, i] == 2:
                edges.append(self.get_edge(node, node2))
            else:
                if self.graph[j, i] == Endpoint.TAIL_AND_ARROW.value \
                        and self.graph[i, j] == Endpoint.ARROW_AND_ARROW.value:
                    edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.TAIL))
                    edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.ARROW))
                else:
                    if self.graph[j, i] == Endpoint.ARROW_AND_ARROW.value \
                            and self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value:
                        edges.append(Edge(node, node2, Endpoint.TAIL, Endpoint.ARROW))
                        edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.ARROW))
                    else:
                        if self.graph[j, i] == Endpoint.TAIL_AND_ARROW.value \
                                and self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value:
                            edges.append(Edge(node, node2, Endpoint.TAIL, Endpoint.TAIL))
                            edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.ARROW))

        return edges

    def get_graph_edges(self) -> List[Edge]:
        edges: List[Edge] = []
        for i in range(self.num_vars):
            node = self.nodes[i]
            for j in range(i + 1, self.num_vars):
                node2 = self.nodes[j]
                if self.graph[j, i] == 1 or self.graph[j, i] == -1 or self.graph[j, i] == 2:
                    edges.append(self.get_edge(node, node2))
                else:
                    if self.graph[j, i] == Endpoint.TAIL_AND_ARROW.value \
                            and self.graph[i, j] == Endpoint.ARROW_AND_ARROW.value:
                        edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.TAIL))
                        edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.ARROW))
                    else:
                        if self.graph[j, i] == Endpoint.ARROW_AND_ARROW.value \
                                and self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value:
                            edges.append(Edge(node, node2, Endpoint.TAIL, Endpoint.ARROW))
                            edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.ARROW))
                        else:
                            if self.graph[j, i] == Endpoint.TAIL_AND_ARROW.value \
                                    and self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value:
                                edges.append(Edge(node, node2, Endpoint.TAIL, Endpoint.TAIL))
                                edges.append(Edge(node, node2, Endpoint.ARROW, Endpoint.ARROW))

        return edges

    # Returns the endpoint along the edge from node1 to node2, at the node2 end.
    def get_endpoint(self, node1: Node, node2: Node) -> Endpoint | None:
        edge = self.get_edge(node1, node2)
        if edge:
            return edge.get_proximal_endpoint(node2)
        else:
            return None

    # Returns true if node2 is a definite noncollider between node1 and node3.
    def is_def_noncollider(self, node1: Node, node2: Node, node3: Node) -> bool:
        edges = self.get_node_edges(node2)
        circle12 = False
        circle23 = False

        for edge in edges:
            _node1 = edge.get_distal_node(node2) == node1
            _node3 = edge.get_distal_node(node2) == node3

            if _node1 and edge.points_toward(node1):
                return True
            if _node3 and edge.points_toward(node3):
                return True

            if _node1 and edge.get_proximal_endpoint(node2) == Endpoint.CIRCLE:
                circle12 = True
            if _node3 and edge.get_proximal_endpoint(node2) == Endpoint.CIRCLE:
                circle23 = True
            if circle12 and circle23 and not self.is_adjacent_to(node1, node2):
                return True

        return False

    # Returns true if node2 is a definite collider between node1 and node3.
    def is_def_collider(self, node1: Node, node2: Node, node3: Node) -> bool:
        edge1 = self.get_edge(node1, node2)
        edge2 = self.get_edge(node2, node3)

        if edge1 is None or edge2 is None:
            return False

        return str(edge1.get_proximal_endpoint(node2)) == "ARROW" and str(edge2.get_proximal_endpoint(node2)) == "ARROW"

    def is_def_unshielded_collider(self, node1: Node, node2: Node, node3: Node) -> bool:
        return self.is_def_collider(node1, node2, node3) and not self.is_directly_connected_to(node1, node3)

    # Returns true if node1 and node2 are d-connected on the set of nodes z.
    def is_dconnected_to(self, node1: Node, node2: Node, z: List[Node]) -> bool:
        utils = GraphUtils()
        return utils.is_dconnected_to(node1, node2, z, self)

    # Returns true if node1 and node2 are d-separated on the set of nodes z.
    def is_dseparated_from(self, node1: Node, node2: Node, z: List[Node]) -> bool:
        return not self.is_dconnected_to(node1, node2, z)

    # Returns true if the graph is a pattern.
    def is_pattern(self) -> bool:
        return self.pattern

    def set_pattern(self, pat: bool):
        self.pattern = pat

    # Returns true if the graph is a PAG.
    def is_pag(self) -> bool:
        return self.pag

    def set_pag(self, pag: bool):
        self.pag = pag

    # Returns true iff there is a single directed edge from node1 to node2.
    def is_directed_from_to(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]

        return self.graph[j, i] == 1 and self.graph[i, j] == -1

    # Returns true iff there is a single undirected edge between node1 and node2.
    def is_undirected_from_to(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]

        return self.graph[j, i] == -1 and self.graph[i, j] == -1

    def is_directly_connected_to(self, node1: Node, node2: Node) -> bool:
        i = self.node_map[node1]
        j = self.node_map[node2]

        return not (self.graph[j, i] == 0 and self.graph[i, j] == 0)

    # Returns true iff the given node is exogenous.
    def is_exogenous(self, node: Node) -> bool:
        return self.get_indegree(node) == 0

    # Returns the nodes adjacent to the given node with the given proximal endpoint.
    def get_nodes_into(self, node: Node, endpoint: Endpoint) -> List[Node]:
        i = self.node_map[node]
        nodes: List[Node] = []

        if str(endpoint) == "ARROW":
            for j in range(self.num_vars):
                if self.graph[i, j] == 1 or self.graph[i, j] == Endpoint.ARROW_AND_ARROW.value:
                    node2 = self.nodes[j]
                    nodes.append(node2)
        else:
            if str(endpoint) == "TAIL":
                for j in range(self.num_vars):
                    if self.graph[i, j] == -1 or self.graph[i, j] == Endpoint.TAIL_AND_ARROW.value:
                        node2 = self.nodes[j]
                        nodes.append(node2)
            else:
                if str(endpoint) == "CIRCLE":
                    for j in range(self.num_vars):
                        if self.graph[i, j] == 2:
                            node2 = self.nodes[j]
                            nodes.append(node2)

        return nodes

    # Returns the nodes adjacent to the given node with the given distal endpoint.
    def get_nodes_out_of(self, node: Node, endpoint: Endpoint) -> List[Node]:
        i = self.node_map[node]
        nodes: List[Node] = []

        if str(endpoint) == "ARROW":
            for j in range(self.num_vars):
                if self.graph[j, i] == 1 or self.graph[j, i] == Endpoint.ARROW_AND_ARROW.value:
                    node2 = self.nodes[j]
                    nodes.append(node2)
        else:
            if str(endpoint) == "TAIL":
                for j in range(self.num_vars):
                    if self.graph[j, i] == -1 or self.graph[j, i] == Endpoint.TAIL_AND_ARROW.value:
                        node2 = self.nodes[j]
                        nodes.append(node2)
            else:
                if str(endpoint) == "CIRCLE":
                    for j in range(self.num_vars):
                        if self.graph[j, i] == 2:
                            node2 = self.nodes[j]
                            nodes.append(node2)

        return nodes

    # Removes the given edge from the graph.
    def remove_edge(self, edge: Edge):
        node1 = edge.get_node1()
        node2 = edge.get_node2()

        i = self.node_map[node1]
        j = self.node_map[node2]

        out_of = self.graph[j, i]
        in_to = self.graph[i, j]

        end1 = edge.get_numerical_endpoint1()
        end2 = edge.get_numerical_endpoint2()

        is_fully_directed = self.is_parent_of(node1, node2) or self.is_parent_of(node2, node1)

        if out_of == Endpoint.TAIL_AND_ARROW.value and in_to == Endpoint.TAIL_AND_ARROW.value:
            if end1 == Endpoint.ARROW.value:
                self.graph[j, i] = -1
                self.graph[i, j] = -1
            else:
                if end1 == -1:
                    self.graph[i, j] = Endpoint.ARROW.value
                    self.graph[j, i] = Endpoint.ARROW.value
        else:
            if out_of == Endpoint.ARROW_AND_ARROW.value and in_to == Endpoint.TAIL_AND_ARROW.value:
                if end1 == Endpoint.ARROW.value:
                    self.graph[j, i] = 1
                    self.graph[i, j] = -1
                else:
                    if end1 == -1:
                        self.graph[j, i] = Endpoint.ARROW.value
                        self.graph[i, j] = Endpoint.ARROW.value
            else:
                if out_of == Endpoint.TAIL_AND_ARROW.value and in_to == Endpoint.ARROW_AND_ARROW.value:
                    if end1 == Endpoint.ARROW.value:
                        self.graph[j, i] = -1
                        self.graph[i, j] = 1
                    else:
                        if end1 == -1:
                            self.graph[j, i] = Endpoint.ARROW.value
                            self.graph[i, j] = Endpoint.ARROW.value
                else:
                    if end1 == in_to and end2 == out_of:
                        self.graph[j, i] = 0
                        self.graph[i, j] = 0

        if is_fully_directed:
            self.reconstitute_dpath(self.get_graph_edges())

    # Removes the edge connecting the given two nodes, provided there is exactly one such edge.
    def remove_connecting_edge(self, node1: Node, node2: Node):
        i = self.node_map[node1]
        j = self.node_map[node2]

        self.graph[j, i] = 0
        self.graph[i, j] = 0

    # Removes all edges connecting node A to node B.  In most cases, this will
    # remove at most one edge, but since multiple edges are permitted in some
    # graph implementations, the number will in some cases be greater than
    # one.
    def remove_connecting_edges(self, node1: Node, node2: Node):
        i = self.node_map[node1]
        j = self.node_map[node2]

        self.graph[j, i] = 0
        self.graph[i, j] = 0

    # Iterates through the list and removes any permissible edges found.  The
    # order in which edges are removed is the order in which they are presented
    # in the iterator.
    def remove_edges(self, edges: List[Edge]):
        for edge in edges:
            self.remove_edge(edge)

    # Removes a node from the graph.
    def remove_node(self, node: Node):
        i = self.node_map[node]

        graph = self.graph

        graph = np.delete(graph, i, axis=0)
        graph = np.delete(graph, i, axis=1)

        self.graph = graph

        nodes = self.nodes
        nodes.remove(node)
        self.nodes = nodes

        # Node indices in node_map should be updated, so rebuild the node_map by the new nodes list
        node_map = {}
        for i, node in enumerate(self.nodes):
            node_map[node] = i
        self.node_map = node_map

        # num_vars should minus 1
        self.num_vars -= 1

        self.reconstitute_dpath(self.get_graph_edges())

    # Iterates through the list and removes any permissible nodes found.  The
    # order in which nodes are removed is the order in which they are presented
    # in the iterator.
    def remove_nodes(self, nodes: List[Node]):
        for node in nodes:
            self.remove_node(node)

    # Constructs and returns a subgraph consisting of a given subset of the
    # nodes of this graph together with the edges between them.
    def subgraph(self, nodes: List[Node]):
        subgraph = GeneralGraph(nodes)
    
        graph = self.graph
    
        nodes_to_delete = []
    
        for i in range(self.num_vars):
            if not (self.nodes[i] in nodes):
                nodes_to_delete.append(i)
    
        graph = np.delete(graph, nodes_to_delete, axis = 0)
        graph = np.delete(graph, nodes_to_delete, axis = 1)
    
        subgraph.graph = graph
        subgraph.reconstitute_dpath(subgraph.get_graph_edges())
    
        return subgraph

    # Returns a string representation of the graph.
    def __str__(self):
        utils = GraphUtils()
        return utils.graph_string(self)

    # Transfers nodes and edges from one graph to another.  One way this is
    # used is to change graph types.  One constructs a new graph based on the
    # old graph, and this method is called to transfer the nodes and edges of
    # the old graph to the new graph.
    def transfer_nodes_and_edges(self, graph):
        for node in graph.nodes:
            self.add_node(node)

        for edge in graph.get_graph_edges():
            self.add_edge(edge)

    def transfer_attributes(self, graph):
        graph.attributes = self.attributes

    # Returns the list of ambiguous triples associated with this graph. Triples <x, y, z> that no longer
    # lie along a path in the getModel graph are removed.
    def get_ambiguous_triples(self) -> List[Tuple[Node, Node, Node]]:
        return self.ambiguous_triples

    # Returns the set of underlines associated with this graph.
    def get_underlines(self) -> List[Tuple[Node, Node, Node]]:
        return self.underline_triples

    # Returns the set of dotted underlines associated with this graph.
    def get_dotted_underlines(self) -> List[Tuple[Node, Node, Node]]:
        return self.dotted_underline_triples

    # Returns true iff the triple <node1, node2, node3> is set as ambiguous.
    def is_ambiguous_triple(self, node1: Node, node2: Node, node3: Node) -> bool:
        return (node1, node2, node3) in self.ambiguous_triples

    # Returns true iff the triple <node1, node2, node3> is set as underlined.
    def is_underline_triple(self, node1: Node, node2: Node, node3: Node) -> bool:
        return (node1, node2, node3) in self.underline_triples

    # Returns true iff the triple <node1, node2, node3> is set as dotted underlined.
    def is_dotted_underline_triple(self, node1: Node, node2: Node, node3: Node) -> bool:
        return (node1, node2, node3) in self.dotted_underline_triples

    # Adds the triple <node1, node2, node3> as an ambiguous triple to the graph.
    def add_ambiguous_triple(self, node1: Node, node2: Node, node3: Node):
        self.ambiguous_triples.append((node1, node2, node3))

    # Adds the triple <node1, node2, node3> as an underlined triple to the graph.
    def add_underline_triple(self, node1: Node, node2: Node, node3: Node):
        self.underline_triples.append((node1, node2, node3))

    # Adds the triple <node1, node2, node3> as a dotted underlined triple to the graph.
    def add_dotted_underline_triple(self, node1: Node, node2: Node, node3: Node):
        self.dotted_underline_triples.append((node1, node2, node3))

    # Removes the triple <node1, node2, node3> from the set of ambiguous triples.
    def remove_ambiguous_triple(self, node1: Node, node2: Node, node3: Node):
        self.ambiguous_triples.remove((node1, node2, node3))

    # Removes the triple <node1, node2, node3> from the set of underlined triples.
    def remove_underline_triple(self, node1: Node, node2: Node, node3: Node):
        self.underline_triples.remove((node1, node2, node3))

    # Removes the triple <node1, node2, node3> from the set of dotted underlined triples.
    def remove_dotted_underline_triple(self, node1: Node, node2: Node, node3: Node):
        self.dotted_underline_triples.remove((node1, node2, node3))

    # Sets the list of ambiguous triples to the triples in the given set.
    def set_ambiguous_triples(self, triples: List[Tuple[Node, Node, Node]]):
        self.ambiguous_triples = triples

    # Sets the list of underlined triples to the triples in the given set.
    def set_underline_triples(self, triples: List[Tuple[Node, Node, Node]]):
        self.underline_triples = triples

    # Sets the list of dotted underlined triples to the triples in the given set.
    def set_dotted_underline_triples(self, triples: List[Tuple[Node, Node, Node]]):
        self.dotted_underline_triples = triples

    # Returns a tier ordering for acyclic graphs.
    def get_causal_ordering(self) -> List[Node]:
        utils = GraphUtils()
        return utils.get_causal_order(self)

    # Returns true if the given node is parameterizable.
    def is_parameterizable(self, node: Node) -> bool:
        return True

    # Returns true if this is a time lag model.
    def is_time_lag_model(self) -> bool:
        return False

    # Returns the nodes in the sepset of node1 and node2.
    def get_sepset(self, node1: Node, node2: Node) -> List[Node]:
        utils = GraphUtils()
        return utils.get_sepset(node1, node2, self)

    # Sets the list of nodes for this graph.
    def set_nodes(self, nodes: List[Node]):
        if len(nodes) != self.num_vars:
            raise ValueError("Sorry, there is a mismatch in the number of variables you are trying to set.")

        self.nodes = nodes

    def get_all_attributes(self):
        return self.attributes

    def get_attribute(self, key):
        return self.attributes[key]

    def remove_attribute(self, key):
        self.attributes.pop[key]  # it's useful

    def add_attribute(self, key, value):
        self.attributes[key] = value

    def get_node_map(self) -> Dict[Node, int]:
        return self.node_map
