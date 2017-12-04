from collections import namedtuple, OrderedDict

import numpy as np
from scipy import linalg

from glycan_profiling.database.composition_network import (
    NeighborhoodWalker, CompositionGraphNode,
    GlycanComposition, GlycanCompositionProxy)

from .constants import DEFAULT_LAPLACIAN_REGULARIZATION, NORMALIZATION
from .graph import network_indices, BlockLaplacian


class LaplacianSmoothingModel(object):
    def __init__(self, network, belongingness_matrix, threshold,
                 regularize=DEFAULT_LAPLACIAN_REGULARIZATION, neighborhood_walker=None,
                 belongingness_normalization=NORMALIZATION):
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

        self.S0 = [node.score for node in self.network[self.obs_ix]]
        self.A0 = belongingness_matrix[self.obs_ix, :]
        self.Am = belongingness_matrix[self.miss_ix, :]
        self.C0 = [node for node in self.network[self.obs_ix]]

        self._belongingness_normalization = belongingness_normalization
        self.variance_matrix = np.eye(len(self.S0))

    def __reduce__(self):
        return self.__class__, (
            self.network, self.belongingness_matrix, self.threshold,
            self.block_L.regularize, self.neighborhood_walker,
            self._belongingness_normalization)

    @property
    def L_mm_inv(self):
        return self.block_L.L_mm_inv

    @property
    def L_oo_inv(self):
        return self.block_L.L_oo_inv

    def optimize_observed_scores(self, lmda, t0=0):
        blocks = self.block_L
        L = lmda * (blocks["oo"] - blocks["om"].dot(self.L_mm_inv).dot(blocks["mo"]))
        B = np.eye(len(self.S0)) + L
        return np.linalg.inv(B).dot(self.S0 - t0) + t0

    def compute_missing_scores(self, observed_scores, t0=0., tm=0.):
        blocks = self.block_L
        return -linalg.inv(blocks['mm']).dot(blocks['mo']).dot(observed_scores - t0) + tm

    def compute_projection_matrix(self, lmbda):
        A = np.eye(self.L_oo_inv.shape[0]) + self.L_oo_inv * (1. / lmbda)
        H = np.linalg.pinv(A)
        return H

    def compute_press(self, observed, updated, projection_matrix):
        press = np.sum(((observed - updated) / (
            1 - (np.diag(projection_matrix) - np.finfo(float).eps))) ** 2) / len(observed)
        return press

    def estimate_tau_from_S0(self, rho, lmda, sigma2=1.0):
        X = ((rho / sigma2) * self.variance_matrix) + (
            (1. / (lmda * sigma2)) * self.L_oo_inv) + self.A0.dot(self.A0.T)
        X = np.linalg.pinv(X)
        return self.A0.T.dot(X).dot(self.S0)

    def get_belongingness_patch(self):
        updated_belongingness = BelongingnessMatrixPatcher.patch(self)
        updated_belongingness = ProportionMatrixNormalization.normalize(
            updated_belongingness, self._belongingness_normalization)
        return updated_belongingness

    def apply_belongingness_patch(self):
        updated_belongingness = self.get_belongingness_patch()
        self.belongingness_matrix = updated_belongingness
        self.A0 = self.belongingness_matrix[self.obs_ix, :]


class ProportionMatrixNormalization(object):
    def __init__(self, matrix):
        self.matrix = np.array(matrix)

    def normalize_columns(self):
        self.matrix = self.matrix / self.matrix.sum(axis=0)

    def normalize_rows(self):
        self.matrix = self.matrix / self.matrix.sum(axis=1).reshape((-1, 1))

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
        elif method.startswith("col") and method[3:4].isdigit():
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
