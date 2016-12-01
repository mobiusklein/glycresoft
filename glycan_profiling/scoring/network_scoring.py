
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

    def update_network(self, base_coef=0.8, support_coef=0.2, iterations=1, **kwargs):
        cg = self.network

        for i in range(iterations):
            if i > 0:
                for node in cg.nodes:
                    node._temp_score = node.score
            else:
                for node in cg.nodes:
                    node._temp_score = node.internal_score

            for node in cg.nodes:
                node.score = self.compute_support(
                    node, base_coef, support_coef, **kwargs)

    def compute_support(self, node, base_coef=0.8, support_coef=0.2, verbose=False, **kwargs):
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


class DistortedNetworkScoreDistributor(NetworkScoreDistributorBase):

    def compute_support(self, node, base_coef=0.8, support_coef=0.2, verbose=False, **kwargs):
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


def nullcallback(net, i):
    pass


nullcallback.complete = lambda net: net


class RenormalizingNetworkScorer(NetworkScoreDistributorBase):

    def compute_support(self, node, base_coef=0.8, support_coef=0.2, verbose=False, **kwargs):
        base = node._temp_score
        support = 0
        n_edges = 0.
        for edge in node.edges:
            other = edge[node]
            if other._temp_score < 0.3:
                continue
            distance = 1. / edge.order
            support += other._temp_score * distance
            n_edges += distance
            if verbose:
                print(other._temp_score, support)
        if n_edges == 0:
            return base
        return base + support_coef * (support / n_edges)

    def normalize(self, original_maximum):
        max_score = max(node.score for node in self.network)
        for node in self.network:
            node.score /= max_score
            node.score *= original_maximum

    def update_network(self, base_coef=0.8, support_coef=0.2, iterations=1, **kwargs):
        cg = self.network
        callback = kwargs.pop("callback", nullcallback)

        if iterations == 1:
            support_coef /= 20.
            iterations *= 20

        for i in range(iterations):
            if i > 0:
                for node in cg.nodes:
                    node._temp_score = node.score
            else:
                for node in cg.nodes:
                    node._temp_score = node.internal_score

            original_maximum = max(node._temp_score for node in cg.nodes)

            for node in cg.nodes:
                node.score = self.compute_support(
                    node, base_coef, support_coef, **kwargs)
            self.normalize(original_maximum)
            callback(self.network, i)

        completed = getattr(callback, "complete", None)
        if completed is not None:
            completed(self.network)


class RenormalizingNetworkScorerDifferenceMethod(RenormalizingNetworkScorer):

    def compute_support(self, node, base_coef=0.8, support_coef=0.2, verbose=False, **kwargs):
        base = node._temp_score
        support = 0
        for edge in node.edges:
            other = edge[node]
            distance = 1. / edge.order
            if other._temp_score > node._temp_score:
                support += (other._temp_score - node._temp_score) * distance
                if verbose:
                    print(other._temp_score, support)
        return base + support_coef * (support)


class RenormalizingNetworkScorerAverageDifferenceMethod(RenormalizingNetworkScorer):

    def compute_support(self, node, base_coef=0.8, support_coef=0.2, verbose=False, **kwargs):
        base = node._temp_score
        support = 0
        n_edges = 0.
        for edge in node.edges:
            other = edge[node]
            distance = 1. / edge.order
            if other._temp_score > node._temp_score:
                support += (other._temp_score - node._temp_score) * distance
                n_edges += distance
                if verbose:
                    print(other._temp_score, support)
        if n_edges == 0:
            return base
        return base + support_coef * (support / n_edges)


NetworkScoreDistributor = RenormalizingNetworkScorerDifferenceMethod
