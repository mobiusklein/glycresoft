from glycresoft.database.builder.glycan import TextFileGlycanHypothesisSerializer
from glycresoft.database.prebuilt.utils import hypothesis_register, BuildBase
from io import StringIO


hypothesis_metadata = {
    "name": "Glycosaminoglycan Linkers",
    "hypothesis_type": "glycan_composition",
    "description": ''
}

source_text = u'''{Xyl:1; a,enHex:1; Hex:2; aHex:1; HexNAc:1}      GAG-linker
{Xyl:1; a,enHex:1; Hex:2; aHex:1; HexNAc(S):1}   GAG-linker
{Xyl:1; Fuc:1; a,enHex:1; Hex:2; aHex:1; HexNAc:1}       GAG-linker
{Xyl:1; Fuc:1; a,enHex:1; Hex:1; aHex:1; HexS:1; HexNAc(S):1}    GAG-linker
{Xyl:1; Fuc:1; a,enHex:1; Hex:2; aHex:1; HexNAc(S):1}    GAG-linker
{Fuc:1; a,enHex:1; Hex:2; aHex:1; HexNAc:1; XylP:1}      GAG-linker
{a,enHex:1; Hex:2; aHex:1; HexNAc:1; XylP:1}     GAG-linker
{a,enHex:1; Hex:2; aHex:1; XylP:1; HexNAc(S):1}  GAG-linker
{a,enHex:1; Hex:1; aHex:1; XylP:1; HexS:1; HexNAc(S):1}  GAG-linker
{Xyl:1; a,enHex:1; Hex:2; aHex:1; HexNAc(S):1}   GAG-linker
{Xyl:1; a,enHex:1; Hex:1; aHex:1; HexS:1; HexNAc(S):1}   GAG-linker
{Xyl:1; a,enHex:1; aHex:1; HexS:2; HexNAc(S):1}  GAG-linker
{Xyl:1; a,enHex:1; Hex:2; aHex:1; HexNAc:1; Neu5Ac:1}    GAG-linker
{Xyl:1; a,enHex:1; Hex:2; aHex:1; HexNAc(S):1; Neu5Ac:1}         GAG-linker
{Xyl:1; a,enHex:1; Hex:1; aHex:1; HexS:1; HexNAc(S):1; Neu5Ac:1}         GAG-linker
{Xyl:1; a,enHex:1; Hex:2; aHex:1; HexNAc:1; Neu5Ac:1}    GAG-linker
{Xyl:1; a,enHex:1; Hex:2; aHex:1; HexNAc(S):1; Neu5Ac:1}         GAG-linker
{a,enHex:1; Hex:2; aHex:1; XylP:1; HexNAc(S):1; Neu5Ac:1}        GAG-linker
{Xyl:1; a,enHex:1; Hex:1; aHex:1; HexS:1; HexNAc(S):1; Neu5Ac:1}         GAG-linker
{a,enHex:1; Hex:1; aHex:1; XylP:1; HexS:1; HexNAc(S):1; Neu5Ac:1}        GAG-linker
{Xyl:1; a,enHex:1; Hex:2; aHex:1; HexNAc:1; Neu5Gc:1}    GAG-linker
{Xyl:1; a,enHex:1; Hex:2; aHex:1; HexNAc(S):1; Neu5Gc:1}         GAG-linker
{Xyl:1; a,enHex:1; Hex:1; aHex:1; HexS:1; HexNAc(S):1; Neu5Gc:1}         GAG-linker
{Xyl:1; a,enHex:1; Hex:2; aHex:1; HexNAc:1; Neu5Gc:1}    GAG-linker
{Xyl:1; a,enHex:1; Hex:2; aHex:1; HexNAc(S):1; Neu5Gc:1}         GAG-linker
{a,enHex:1; Hex:2; aHex:1; XylP:1; HexNAc(S):1; Neu5Gc:1}        GAG-linker
{Xyl:1; a,enHex:1; Hex:1; aHex:1; HexS:1; HexNAc(S):1; Neu5Gc:1}         GAG-linker
{a,enHex:1; Hex:1; aHex:1; XylP:1; HexS:1; HexNAc(S):1; Neu5Gc:1}        GAG-linker'''


@hypothesis_register(hypothesis_metadata['name'])
class GlycosaminoglycanLinkersBuilder(BuildBase):

    def get_hypothesis_metadata(self):
        return hypothesis_metadata

    def prepare_buffer(self):
        text_buffer = StringIO(source_text)
        return text_buffer

    def build(self, database_connection, **kwargs):
        kwargs.setdefault('hypothesis_name', self.hypothesis_metadata['name'])
        task = TextFileGlycanHypothesisSerializer(
            self.prepare_buffer(), database_connection,
            **kwargs)
        task.start()
        return task
