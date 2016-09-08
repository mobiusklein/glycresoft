
class NetworkScoreDistributor(object):
    def __init__(self, solutions, network):
        self.solutions = solutions
        self.network = network

    def build_solution_map(self):
        self.solution_map = {
            sol.chromatogram.composition: sol
            for sol in self.solutions
            if sol.chromatogram.composition is not None
        }
        return self.solution_map

    def _set_up_temporary_score(self, items, iteration=0):
        if iteration > 0:
            for sol in items:
                sol._temp_score = sol.score
        else:
            for sol in items:
                sol._temp_score = sol.internal_score

    def score_solution(self, solution, base_coef=0.8, support_coef=0.2):
        sol = solution
        base = base_coef * sol._temp_score
        support = 0
        n_edges_matched = 0
        if sol.composition is not None:
            cn = self.network[sol.composition]
            for edge in cn.edges:
                other = edge[cn]
                if other in self.solution_map:
                    n_edges_matched += 1
                    other_sol = self.solution_map[other]
                    support += support_coef * edge.weight * other_sol._temp_score
            sol.score = base + min(support, support_coef)
        else:
            sol.score = base
        return base, support, sol.score, n_edges_matched

    def distribute(self, base_coef=0.8, support_coef=0.2, iterations=1):
        self.build_solution_map()
        for i in range(iterations):
            self._set_up_temporary_score(self.solutions, i)
            for sol in self.solutions:
                self.score_solution(sol, base_coef, support_coef)

    def assign_network(self):
        solution_map = self.build_solution_map()

        cg = self.network

        for node in cg.nodes:
            node.internal_score = 0
            node._temp_score = 0

        for composition, solution in solution_map.items():
            node = cg[composition]
            node.internal_score = solution.score

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
                base = node._temp_score * base_coef
                support = 0
                for edge in node.edges:
                    other = edge[node]
                    support += support_coef * edge.weight * other._temp_score
                node.score = min(base + (support), 1.0)
