
class NetworkScoreDistributorBase(object):
    def __init__(self, solutions, network):
        self.solutions = solutions
        self.network = network

    def build_solution_map(self):
        self.solution_map = {
            sol.chromatogram.glycan_composition: sol
            for sol in self.solutions
            if sol.chromatogram.glycan_composition is not None
        }
        return self.solution_map

    def _set_up_temporary_score(self, items, iteration=0):
        if iteration > 0:
            for sol in items:
                sol._temp_score = sol.score
        else:
            for sol in items:
                sol._temp_score = sol.internal_score

    def assign_network(self):
        solution_map = self.build_solution_map()

        cg = self.network

        for node in cg.nodes:
            node.internal_score = 0
            node._temp_score = 0

        for composition, solution in solution_map.items():
            node = cg[composition]
            node._temp_score = node.internal_score = solution.internal_score

    def update_network(self, base_coef=0.8, support_coef=0.2, iterations=1):
        cg = self.network

        for i in range(iterations):
            if i > 0:
                for node in cg.nodes:
                    node._temp_score = node.score
            else:
                for node in cg.nodes:
                    node._temp_score = node.internal_score

            for node in cg.nodes:
                node.score = self.compute_support(node, base_coef, support_coef)

    def compute_support(self, node, base_coef=0.8, support_coef=0.2, verbose=False):
        base = node._temp_score * base_coef
        support = 0
        n_edges = 0.
        for edge in node.edges:
            other = edge[node]
            if other._temp_score < 0.5:
                continue
            support += support_coef * edge.weight * other._temp_score
            n_edges += edge.weight
            if verbose:
                print(other._temp_score, support)
        return min(base + (support / n_edges), 1.0)

    def update_solutions(self):
        for node in self.network:
            if node.glycan_composition in self.solution_map:
                sol = self.solution_map[node.glycan_composition]
                sol.score = node.score

    def distribute(self, base_coef=0.8, support_coef=0.2):
        self.build_solution_map()
        self.assign_network()
        self.update_network(base_coef, support_coef)
        self.update_solutions()


class NetworkScoreDistributor(NetworkScoreDistributorBase):

    def compute_support(self, node, base_coef=0.8, support_coef=0.2, verbose=False):
        base = node._temp_score * base_coef
        support = 0
        weights = 0
        for edge in node.edges:
            other = edge[node]
            if other._temp_score < 0.5:
                continue
            distance = 1. / edge.order
            support += edge.weight * (other._temp_score ** 2 * 10)
            weights += edge.weight * distance * 7.
            if verbose:
                print(other._temp_score, support, weights)
        if weights == 0:
            weights = 1.0
        return min(base + (support_coef * (support / weights)), 1.0)
