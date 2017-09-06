from glycan_profiling.database.builder.glycan import glycosaminoglycan_linker_generation
from glycan_profiling.database.builder.glycan import TextFileGlycanHypothesisSerializer
from glycan_profiling.database.prebuilt.utils import hypothesis_register, BuildBase
from io import StringIO


hypothesis_metadata = {
    "name": "Glycosaminoglycan Linkers",
    "hypothesis_type": "glycan_composition"
}


@hypothesis_register(hypothesis_metadata['name'])
class GlycosaminoglycanLinkersBuilder(BuildBase):

    def get_hypothesis_metadata(self):
        return hypothesis_metadata

    def prepare_buffer(self):
        text_buffer = StringIO()
        for glycan_composition in glycosaminoglycan_linker_generation.gag_linker_compositions():
            text_buffer.write(u"{}\tGAG-Linker\n".format(glycan_composition.serialize()))
        text_buffer.seek(0)
        return text_buffer

    def build(self, database_connection, **kwargs):
        kwargs.setdefault('hypothesis_name', self.hypothesis_metadata['name'])
        task = TextFileGlycanHypothesisSerializer(
            self.prepare_buffer(), database_connection,
            **kwargs)
        task.start()
