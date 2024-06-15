# -*- coding: utf-8 -*-

from collections import namedtuple, OrderedDict
from typing import List, Tuple

from six import string_types as basestring

import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

from glypy import GlycanComposition
from glycopeptidepy.structure.glycan import GlycanCompositionProxy

from glycresoft.database.composition_network import (
    NeighborhoodWalker, CompositionGraphNode, CompositionGraph)

from .constants import DEFAULT_LAPLACIAN_REGULARIZATION, NORMALIZATION
from .graph import network_indices, BlockLaplacian


class LaplacianSmoothingModel(object):
    r'''A model that uses a graphical representation of a set of related entities
    to estimate a connectedness-aware version of a quality score :math:`s`.

    The `Graph Laplacian <https://en.wikipedia.org/wiki/Laplacian_matrix>`_ is used
    to summarize the connectedness of nodes in the graph.

    Related subgraphs are further organized into neighborhoods, as defined by
    :attr:`neighborhood_walker`'s rules. These groups are expected to trend together,
    sharing a parameter :math:`\tau`. A single vertex in the graph can belong to multiple
    neighborhoods to different extents, having a belongingness weight for each neighborhood
    encoded in a belongingness matrix :math:`\mathbf{A}`.
    '''

    network: CompositionGraph
    neighborhood_walker: NeighborhoodWalker

    belongingness_matrix: np.ndarray
    variance_matrix: np.ndarray
    inverse_variance_matrix: np.ndarray

    block_L: BlockLaplacian
    obs_ix: np.ndarray
    miss_ix: np.ndarray

    S0: np.ndarray
    A0: np.ndarray
    Am: np.ndarray

    C0: List[CompositionGraphNode]

    threshold: float
    regularize: float

    _belongingness_normalization: str

    def __init__(self, network, belongingness_matrix, threshold,
                 regularize=DEFAULT_LAPLACIAN_REGULARIZATION, neighborhood_walker=None,
                 belongingness_normalization=NORMALIZATION, variance_matrix=None,
                 inverse_variance_matrix=None):
        self.network = network

        if neighborhood_walker is None:
            self.neighborhood_walker = NeighborhoodWalker(self.network)
        else:
            self.neighborhood_walker = neighborhood_walker

        self.belongingness_matrix = belongingness_matrix
        self.threshold = threshold

        self.obs_ix, self.miss_ix = network_indices(self.network, self.threshold)
        self.block_L = BlockLaplacian(
            self.network, threshold, regularize=regularize)

        self.S0 = np.array([node.score for node in self.network[self.obs_ix]])
        self.A0 = belongingness_matrix[self.obs_ix, :]
        self.Am = belongingness_matrix[self.miss_ix, :]
        self.C0 = [node for node in self.network[self.obs_ix]]

        self._belongingness_normalization = belongingness_normalization
        if variance_matrix is None:
            variance_matrix = np.eye(len(self.S0))
        self.variance_matrix = variance_matrix
        if inverse_variance_matrix is None:
            inverse_variance_matrix = np.eye(len(self.S0))
        self.inverse_variance_matrix = inverse_variance_matrix

    def __reduce__(self):
        return self.__class__, (
            self.network, self.belongingness_matrix, self.threshold,
            self.block_L.regularize, self.neighborhood_walker,
            self._belongingness_normalization, self.variance_matrix,
            self.inverse_variance_matrix)

    @property
    def L_mm_inv(self) -> np.ndarray:
        return self.block_L.L_mm_inv

    @property
    def L_oo_inv(self) -> np.ndarray:
        return self.block_L.L_oo_inv

    def optimize_observed_scores(self, lmda: float, t0: np.ndarray=0) -> np.ndarray:
        blocks = self.block_L
        # Here, V_inv is the inverse of V, which is the inverse of the variance matrix
        V_inv = self.variance_matrix
        L = lmda * V_inv.dot(blocks["oo"] - blocks["om"].dot(self.L_mm_inv).dot(blocks["mo"]))
        B = np.identity(len(self.S0)) + L
        return np.linalg.inv(B).dot(self.S0 - t0) + t0

    def compute_missing_scores(self, observed_scores: np.ndarray, t0=0., tm=0.) -> np.ndarray:
        blocks = self.block_L
        return -linalg.inv(blocks['mm']).dot(blocks['mo']).dot(observed_scores - t0) + tm

    def compute_projection_matrix(self, lmbda) -> np.ndarray:
        A = np.eye(self.L_oo_inv.shape[0]) + self.L_oo_inv * (1. / lmbda)
        H = np.linalg.pinv(A)
        return H

    def compute_press(self, observed: np.ndarray, updated: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
        press = np.sum(((observed - updated) / (
            1 - (np.diag(projection_matrix) - np.finfo(float).eps))) ** 2) / len(observed)
        return press

    def estimate_tau_from_S0(self, rho: float, lmda: float, sigma2: float=1.0) -> np.ndarray:
        X = ((rho / sigma2) * self.variance_matrix) + (
            (1. / (lmda * sigma2)) * self.L_oo_inv) + self.A0.dot(self.A0.T)
        X = np.linalg.pinv(X)
        return self.A0.T.dot(X).dot(self.S0)

    def estimate_phi_given_S_parameters(self, lmbda: float, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Estimate the parameters of the conditional distribution :math:`\phi|s`
        according to [1]_.

        Notes
        -----
        According to [1]_, given
        .. math::

                y | \theta_1 \sim & \mathcal{N}\left(\mathbf{A_1}\theta_1, \mathbf{C_1}\right) \\

                \theta_1 \sim & \mathcal{N}\left(\mathbf{A_2}\theta_2, \mathbf{C_2}\right) \\

                \theta_1 | y \sim & \mathcal{N}\left(\mathbf{B}b, \mathbf{B}\right) \\

                \mathbf{B^{-1}} = & \mathbf{C_2}^{-1} + \mathbf{A_1}^t\mathbf{C_1}^{-1}\mathbf{A_1} \\
                b = & \mathbf{A_1}^t\mathbf{C_1}^{-1}y + \mathbf{C_2}^{-1}\mathbf{A_2}\theta_2

        In terms of the network smoothing model, this translates to

        .. math::

                \mathbf{B^{-1}} = & \lambda\mathbf{L} + \tilde{A}^t\Sigma^{-1}\tilde{A}\\
                                = & \lambda \mathbf{L} + \begin{bmatrix} \Sigma_{oo}^{-1} & 0 \\ 0 & 0 \end{bmatrix} \\

                b = & \tilde{A}^t\Sigma^{-1}s + \lambda\mathbf{L}\mathbf{A}\tau \\
                  = & \begin{bmatrix}\Sigma_{oo}^{-1}s \\ 0\end{bmatrix} + \lambda\begin{bmatrix}L_{oo}A\tau_o +
                      L_{om}A\tau_m \\L_{mo}A\tau_o + L_{mm}A\tau_m\end{bmatrix}\\\\

        Parameters
        ----------
        lmbda : float
            The smoothing factor
        tau : np.ndarray
            The mean of :math:`\phi`, :math:`\mathbf{A}\tau`

        Returns
        -------
        Bb: np.ndarray
            The mean vector of :math:`\phi|s`
        B: np.ndarray:
            The variance matrix of :math:`\phi|s`

        References
        ----------
        .. [1] Lindley, D. V., & Smith, A. F. M. (1972). Bayes Estimates for the Linear Model.
               Royal Statistical Society, 34(1), 1–41.
        """
        B_inv = lmbda * self.block_L.matrix
        B_inv[self.obs_ix, self.obs_ix] += np.diag(self.inverse_variance_matrix)
        B = linalg.inv(B_inv)

        Alts = np.zeros(len(self.network))
        Alts[self.obs_ix] = self.S0.dot(self.inverse_variance_matrix)
        lambda_Lw_Atau = lmbda * self.block_L.matrix.dot(self.normalized_belongingness_matrix.dot(tau))
        b = Alts + lambda_Lw_Atau
        phi_given_s = np.dot(B, b)
        return phi_given_s, B

    def build_belongingness_matrix(self) -> np.ndarray:
        belongingness_matrix = self.neighborhood_walker.build_belongingness_matrix()
        return belongingness_matrix

    def get_belongingness_patch(self) -> np.ndarray:
        updated_belongingness = BelongingnessMatrixPatcher.patch(self)
        updated_belongingness = ProportionMatrixNormalization.normalize(
            updated_belongingness, self._belongingness_normalization)
        return updated_belongingness

    def apply_belongingness_patch(self):
        updated_belongingness = self.get_belongingness_patch()
        self.belongingness_matrix = updated_belongingness
        self.A0 = self.belongingness_matrix[self.obs_ix, :]


class SmoothedScoreSample(object):
    def __init__(self, model, lmbda, tau, size=10000, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState()
        self.model = model
        self.lmbda = lmbda
        self.tau = tau
        self.size = size
        self.Bb, self.B = self.model.estimate_phi_given_S_parameters(lmbda, tau)
        self.B_inv = np.linalg.pinv(self.B)
        self.random_state = random_state
        self.samples = np.array([])
        self.resample()

    def resample(self):
        self.samples = self.random_state.multivariate_normal(self.Bb, self.B, size=self.size)
        # alternatively, sample unit variance scaled by the cholesky decomposition of the covariance
        # matrix to get the sample variance
        # C = linalg.cholesky(self.B)
        # self.samples = np.array(
        #                   [self.Bb + C.T.dot(self.random_state.normal(size=C.shape[0])) for i in range(self.size)])
        return self

    def _get_index(self, glycan_composition):
        node = self.model.network[glycan_composition]
        index = node.index
        return index

    def _score_index(self, index):
        return self.Bb[index]

    def _gaussian_pdf(self, x, mu, sigma, sigma_inv):
        d = (x - mu)
        e = np.exp((-0.5 * (d) * sigma_inv * d))
        return e / np.sqrt(2 * np.pi * sigma)

    def score(self, glycan_composition):
        index = self._get_index(glycan_composition)
        return self._score_index(index)

    def plot_distribution(self, glycan_composition, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        index = self._get_index(glycan_composition)
        node = self.model.network[index]
        s = self.samples[:, index]
        bins = np.arange(0, self.Bb[index] * 2, .1)
        ax.hist(s, bins=bins, alpha=0.5, density=True, label="%s (%f)" % (node, self.score(glycan_composition)))
        ax.set_xlabel(r"Resampled $\phi$", fontsize=24)
        ax.set_ylabel("Density", fontsize=24)
        return ax


class ProportionMatrixNormalization(object):
    def __init__(self, matrix):
        self.matrix = np.array(matrix)

    def normalize_columns(self):
        self.matrix = self.matrix / self.matrix.sum(axis=0)

    def normalize_rows(self):
        normalizer = self.matrix.sum(axis=1).reshape((-1, 1))
        normalizer[normalizer == 0] = 1e-6
        self.matrix = self.matrix / normalizer

    def normalize_columns_and_rows(self):
        self.normalize_columns()
        self.clean()
        self.normalize_rows()

    def normalize_columns_scaled(self, scaler=2.0):
        self.normalize_columns()
        self.matrix = self.matrix * scaler

    def clean(self):
        self.matrix[np.isnan(self.matrix)] = 0.0

    def __array__(self):
        return self.matrix

    @classmethod
    def normalize(cls, matrix, method='colrow'):
        self = cls(matrix)
        if method == 'col':
            self.normalize_columns()
        elif method == 'row':
            self.normalize_rows()
        elif method == 'colrow':
            self.normalize_columns_and_rows()
        elif method is not None and method.startswith("col") and method[3:4].isdigit():
            scale = float(method[3:])
            self.normalize_columns_scaled(scale)
        elif method == 'none' or method is None:
            pass
        else:
            raise ValueError("Unknown Normalization Method %r" % method)
        self.clean()
        return self.matrix


class GroupBelongingnessMatrix(object):
    _node_like = (basestring, CompositionGraphNode,
                  GlycanComposition, GlycanCompositionProxy)

    @classmethod
    def from_model(cls, model, normalized=True):
        if normalized:
            mat = model.normalized_belongingness_matrix
        else:
            mat = model.belongingness_matrix
        groups = model.neighborhood_walker.neighborhood_names()
        members = model.network.nodes
        return cls(mat, groups, members)

    def __init__(self, belongingness_matrix, groups, members):
        self.belongingness_matrix = np.array(belongingness_matrix)
        self.groups = [str(x) for x in groups]
        self.members = [str(x) for x in members]
        self._column_indices = OrderedDict([(k, i) for i, k in enumerate(self.groups)])
        self._member_indices = OrderedDict([(k, i) for i, k in enumerate(self.members)])

    def _column_indices_by_name(self, names):
        if isinstance(names, basestring):
            names = [names]
        indices = [self._column_indices[n] for n in names]
        return indices

    def _member_indices_by_name(self, names):
        if isinstance(names, self._node_like):
            names = [names]
        indices = [self._member_indices[n] for n in names]
        return indices

    def _coerce_member(self, names):
        if isinstance(names, self._node_like):
            names = [names]
        return names

    def _coerce_column(self, names):
        if isinstance(names, basestring):
            names = [names]
        return names

    def get(self, rows=None, cols=None):
        matrix = self.belongingness_matrix
        if rows is not None:
            rows = self._coerce_member(rows)
            row_ix = self._member_indices_by_name(rows)
            matrix = matrix[row_ix, :]
        else:
            rows = self.members
        if cols is not None:
            cols = self._coerce_column(cols)
            col_ix = self._column_indices_by_name(cols)
            matrix = matrix[:, col_ix]
        else:
            cols = self.groups
        return self.__class__(matrix, cols, rows)

    def getindex(self, rows=None, cols=None):
        if rows is not None:
            rows = self._coerce_member(rows)
            row_ix = self._member_indices_by_name(rows)
        else:
            row_ix = None
        if cols is not None:
            cols = self._coerce_column(cols)
            col_ix = self._column_indices_by_name(cols)
        else:
            col_ix = None
        return row_ix, col_ix

    def __getitem__(self, ij):
        return self.belongingness_matrix[ij]

    def __setitem__(self, ij, val):
        self.belongingness_matrix[ij] = val

    def __array__(self):
        return np.array(self.belongingness_matrix)

    def __repr__(self):
        column_label_width = max(map(len, self.groups))
        row_label_width = max(map(len, self.members))
        rows = []
        top_row = [' ' * row_label_width]
        for col in self.groups:
            top_row.append(col.center(column_label_width))
        rows.append('|'.join(top_row))
        for i, member in enumerate(self.members):
            row = [member.ljust(row_label_width)]
            vals = self.belongingness_matrix[i, :]
            for val in vals:
                row.append(("%0.3f" % val).center(column_label_width))
            rows.append("|".join(row))
        return '\n'.join(rows)


MatrixEditIndex = namedtuple("MatrixEditIndex", ("row_index", "col_index", "action"))
MatrixEditInstruction = namedtuple("MatrixEditInstruction", ("composition", "neighborhood", "action"))


class BelongingnessMatrixPatcher(object):
    def __init__(self, model):
        self.model = model
        self.A0 = model.A0
        self.belongingness_matrix = GroupBelongingnessMatrix.from_model(
            model, normalized=False)

    def find_singleton_neighborhoods(self):
        n_cols = self.A0.shape[1]
        edits = []
        for i in range(n_cols):
            # We have a neighborhood with only one member
            col = self.A0[:, i]
            mask = col > 0
            if mask.sum() == 1:
                j = np.argmax(mask)
                edit = MatrixEditIndex(j, i, 'delete')
                edits.append(edit)
        return edits

    def transform_index_to_key(self, edits):
        neighborhood_names = self.model.neighborhood_walker.neighborhood_names()
        out = []
        for edit in edits:
            key = str(self.model.C0[edit.row_index])
            neighborhood = neighborhood_names[edit.col_index]
            out.append(MatrixEditInstruction(key, neighborhood, edit.action))
        return out

    def patch_belongingness_matrix(self, edits):
        gbm = self.belongingness_matrix
        instructions = self.transform_index_to_key(edits)
        for instruction in instructions:
            ij = gbm.getindex(instruction.composition, instruction.neighborhood)
            if instruction.action == 'delete':
                gbm[ij] = 0
        return gbm

    @classmethod
    def patch(cls, model):
        inst = cls(model)
        targets = inst.find_singleton_neighborhoods()
        out = inst.patch_belongingness_matrix(targets)
        return np.array(out)
