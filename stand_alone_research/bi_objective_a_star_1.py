import heapq

class Node:
    def __init__(self, state, cost1=0, cost2=0, heuristic1=0, heuristic2=0, parent=None):
        self.state = state
        self.cost1 = cost1
        self.cost2 = cost2
        self.heuristic1 = heuristic1
        self.heuristic2 = heuristic2
        self.parent = parent

    def __lt__(self, other):
        return (self.f1, self.f2) < (other.f1, other.f2)

    def __eq__(self, other):
        return self.state == other.state

    @property
    def f1(self):
        return self.cost1 + self.heuristic1

    @property
    def f2(self):
        return self.cost2 + self.heuristic2


def reconstruct_path(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]


def boa_star(start, goal, graph, heuristic, eps=0):

    def _is_dominated(_n, _cost_2_min, sols, _eps):
        return _n.cost2 >= _cost_2_min.get(_n.state, float('inf')) or (1 + _eps) * _n.f2 >= min([_cost_2_min[_sol.state] for _sol in sols], default=float('inf'))

    open_list = []
    cost_2_min = {}
    sols = []  # Store Pareto-optimal solutions (Nodes)
    start_node = Node(start, 0, 0, heuristic[start][0], heuristic[start][1])
    heapq.heappush(open_list, start_node)  # should be ordered by (cost_1, cost_2)
    cost_2_min[start_node.state] = 0

    while open_list:
        current_node = heapq.heappop(open_list)

        # is dominated
        if current_node != start_node and _is_dominated(current_node, cost_2_min, sols, eps):
            continue

        cost_2_min[current_node.state] = current_node.cost2

        if goal(current_node.state):
            sols.append(current_node)
            continue

        for neighbor, (c1, c2) in graph.get_neighbours(current_node.state):  # should return a mapping between a neighbour vertex to both edge costs
            new_g1 = current_node.cost1 + c1
            new_g2 = current_node.cost2 + c2
            neighbor_node = Node(neighbor, new_g1, new_g2, heuristic[neighbor][0], heuristic[neighbor][1],
                                 parent=current_node)

            # checking if 'neighbor' is dominated
            if _is_dominated(neighbor_node, cost_2_min, sols, eps):
                continue

            heapq.heappush(open_list, neighbor_node)

    return [reconstruct_path(sol) for sol in sols]  # Return list of paths