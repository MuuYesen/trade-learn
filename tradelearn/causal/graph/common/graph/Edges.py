#!/usr/bin/env python3
from __future__ import annotations

from .Edge import Edge
from .Endpoint import Endpoint
from .Node import Node


class Edges:

    # constructs a new bidirected edge from node_a to node_b <->
    def bidirected_edge(self, node_a: Node, node_b: Node) -> Edge:
        edge = Edge(node_a, node_b, Endpoint.ARROW, Endpoint.ARROW)
        return edge

    # constructs a new directed edge from node_a to node_b -->
    def directed_edge(self, node_a: Node, node_b: Node) -> Edge:
        edge = Edge(node_a, node_b, Endpoint.TAIL, Endpoint.ARROW)
        return edge

    # constructs a new partially oriented edge from node_a to node_b o->
    def partially_oriented_edge(self, node_a: Node, node_b: Node) -> Edge:
        edge = Edge(node_a, node_b, Endpoint.CIRCLE, Endpoint.ARROW)
        return edge

    # constructs a new undirected edge from node_a to node_b --
    def undirected_edge(self, node_a: Node, node_b: Node) -> Edge:
        edge = Edge(node_a, node_b, Endpoint.TAIL, Endpoint.TAIL)
        return edge

    # return true iff an edge is a bidrected edge <->
    def is_bidirected_edge(self, edge: Edge) -> bool:
        return edge.get_endpoint1() == Endpoint.ARROW and edge.get_endpoint2() == Endpoint.ARROW

    # return true iff the given edge is a directed edge -->
    def is_directed_edge(self, edge: Edge) -> bool:
        if edge.get_endpoint1() == Endpoint.TAIL:
            return edge.get_endpoint2() == Endpoint.ARROW
        elif edge.get_endpoint2() == Endpoint.TAIL:
            return edge.get_endpoint1() == Endpoint.ARROW
        else:
            return False

    # return true iff the given edge is a partially oriented edge o->
    def is_partially_oriented_edge(self, edge: Edge) -> bool:
        if edge.get_endpoint1() == Endpoint.CIRCLE:
            return edge.get_endpoint2() == Endpoint.ARROW
        elif edge.get_endpoint2() == Endpoint.CIRCLE:
            return edge.get_endpoint1() == Endpoint.ARROW
        else:
            return False

    # return true iff some edge is an undirected edge --
    def is_undirected_edge(self, edge: Edge) -> bool:
        return edge.get_endpoint1() == Endpoint.TAIL and edge.get_endpoint2() == Endpoint.TAIL

    def traverse_directed(self, node: Node, edge: Edge) -> Node | None:
        if node == edge.get_node1():
            if str(edge.get_endpoint1()) == "TAIL" and str(edge.get_endpoint2()) == "ARROW":
                return edge.get_node2()
        elif node == edge.get_node2():
            if str(edge.get_endpoint2()) == "TAIL" and str(edge.get_endpoint1()) == "ARROW":
                return edge.get_node1()

        return None
