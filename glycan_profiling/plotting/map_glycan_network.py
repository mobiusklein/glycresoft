import numpy as np
from matplotlib import pyplot as plt
from glycan_profiling.composition_distribution_model import laplacian_matrix


def _spectral_layout(network):
    eigvals, eigvects = np.linalg.eig(laplacian_matrix(network))
    index = np.argsort(eigvects)[2:3]
    coords = (eigvects[:, index[0][0]], eigvects[:, index[0][1]])
    return coords


def draw_network(network, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    font_args = {"ha": "center", 'va': 'center', "fontdict": {"family": "monospace"}}

    # Spectral Layout Algorithm
    coords = _spectral_layout(network)

    seen_edges = set()
    for node in network.nodes:
        xc, yc = coords[0][node.index], coords[1][node.index]
        ax.scatter(xc, yc, s=250, alpha=0.5)
        label = str(node)
        label = '\n'.join([t.replace(":", "") for t in label[1:-1].split("; ")])
        ax.text(xc, yc, label, size=10, **font_args)
        for e in node.edges:
            if e in seen_edges:
                continue
            else:
                seen_edges.add(e)
            neighbor = e[node]
            neigh_xc, neigh_yc = coords[0][neighbor.index], coords[1][neighbor.index]
            ax.plot((xc, neigh_xc), (yc, neigh_yc), color='grey', alpha=0.5)
            center_xc = (xc + neigh_xc) / 2.
            center_yc = (yc + neigh_yc) / 2.
            ax.text(center_xc, center_yc, "%0.2f" % (1. / e.order), size=7, **font_args)
    ax.axis("off")
    return ax


class VertexWrapper(object):
    def __init__(self, node, position, dispersion):
        self.node = node
        self.position = position
        self.dispersion = dispersion

    def __eq__(self, other):
        return self.node == other.node

    def __ne__(self, other):
        return self.node != other.node


def _spring_layout(network, width=None, height=None, maxiter=100):
    if width is None:
        width = 500.0
    if height is None:
        height = 500.0
    area = width * height
    area = float(area)
    vertices = []
    X, Y = _spectral_layout(network)
    temperature = 100.0
    for i, node in enumerate(network.nodes):
        vert = VertexWrapper(node, np.array((X[i], Y[i])), 0)
        vertices.append(vert)
    k = np.sqrt(area / i)

    def attract(x):
        return (x * x) / k

    def repel(x):
        return (k * k) / x

    for v in vertices:
        v.dispersion = 0
        for u in vertices:
            if v == u:
                continue
            delta = v.position - u.position
            magdelta = len(delta)
            v.dispersion = v.dispersion + (delta / magdelta) * repel(magdelta)

    for edge in network.edges:
        v = vertices[edge.node1.index]
        u = vertices[edge.node2.index]
        delta = v.position - u.position
        magdelta = len(delta)
        attract_magdelta = attract(magdelta)
        delta_magdelta_ratio = (delta / magdelta)
        dispersion_shift = delta_magdelta_ratio * attract_magdelta

        v.dispersion = v.dispersion - dispersion_shift
        u.dispersion = u.dispersion + dispersion_shift

    for v in vertices:
        v.position = v.position + (v.dispersion / len(v.dispersion)) * np.min([v.dispersion, temperature])
        v.position[0] = np.min([width / 2., np.max([-width / 2., v.position[0]])])
        v.position[1] = np.min([height / 2., np.max([-height / 2., v.position[1]])])
    temperature /= 2.
    X = np.array([v.position[0] for v in vertices])
    Y = np.array([v.position[1] for v in vertices])
    return X, Y
