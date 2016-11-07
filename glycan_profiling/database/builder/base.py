from glycan_profiling.task import TaskBase


class HypothesisSerializerBase(TaskBase):

    def set_parameters(self, params):
        if self.hypothesis.parameters is None:
            self.hypothesis.parameters = {}
        self.hypothesis.parameters.update(params)
        self.session.add(self.hypothesis)
        self.session.commit()

    @property
    def hypothesis(self):
        if self._hypothesis is None:
            self._construct_hypothesis()
        return self._hypothesis

    @property
    def hypothesis_name(self):
        if self._hypothesis_name is None:
            self._construct_hypothesis()
        return self._hypothesis_name

    @property
    def hypothesis_id(self):
        if self._hypothesis_id is None:
            self._construct_hypothesis()
        return self._hypothesis_id
