from glycan_profiling.task import TaskBase


class HypothesisSerializerBase(TaskBase):

    def set_parameters(self, params):
        if self.hypothesis.parameters is None:
            self.hypothesis.parameters = {}
            self.session.add(self.hypothesis)
            self.session.commit()
        self.log("Parameters Before Update (%r)" % (self.hypothesis.parameters,))
        for k, v in params.items():
            self.hypothesis.parameters[k] = v    
        # self.hypothesis.parameters.update(params)
        self.log("Set Parameters (%r)" % (self.hypothesis.parameters))
        self.session.add(self.hypothesis)
        self.session.commit()
        self.log("Updated Parameters (%r)" % (self.hypothesis.parameters,))

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

    def on_end(self):
        hypothesis = self.hypothesis
        self.session.add(hypothesis)
        hypothesis.status = "complete"
        self.session.add(hypothesis)
        self.session.commit()
        self.log("Hypothesis Completed")
