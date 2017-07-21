import unittest
import tempfile

from glycan_profiling.serialize.hypothesis.peptide import Peptide, Protein, Glycopeptide
from glycan_profiling.database.builder.glycopeptide import naive_glycopeptide
from glycan_profiling.database.builder.glycopeptide.proteomics import fasta
from glycan_profiling.database.builder.glycan import (
    TextFileGlycanHypothesisSerializer, CombinatorialGlycanHypothesisSerializer)
from glycan_profiling import serialize
from glycan_profiling.test import fixtures

from glycan_profiling.test.test_constrained_combinatorics import FILE_SOURCE as GLYCAN_RULE_FILE_SOURCE

from glycopeptidepy.structure import modification


FASTA_FILE_SOURCE = """
>sp|P02763|A1AG1_HUMAN Alpha-1-acid glycoprotein 1 OS=Homo sapiens GN=ORM1 PE=1 SV=1
MALSWVLTVLSLLPLLEAQIPLCANLVPVPITNATLDQITGKWFYIASAFRNEEYNKSVQ
EIQATFFYFTPNKTEDTIFLREYQTRQDQCIYNTTYLNVQRENGTISRYVGGQEHFAHLL
ILRDTKTYMLAFDVNDEKNWGLSVYADKPETTKEQLGEFYEALDCLRIPKSDVVYTDWKK
DKCEPLEKQHEKERKQEEGES

>sp|P19652|A1AG2_HUMAN Alpha-1-acid glycoprotein 2 OS=Homo sapiens GN=ORM2 PE=1 SV=2
MALSWVLTVLSLLPLLEAQIPLCANLVPVPITNATLDRITGKWFYIASAFRNEEYNKSVQ
EIQATFFYFTPNKTEDTIFLREYQTRQNQCFYNSSYLNVQRENGTVSRYEGGREHVAHLL
FLRDTKTLMFGSYLDDEKNWGLSFYADKPETTKEQLGEFYEALDCLCIPRSDVMYTDWKK
DKCEPLEKQHEKERKQEEGES
"""

HEPARANASE = """
>sp|Q9Y251|HPSE_HUMAN Heparanase OS=Homo sapiens GN=HPSE PE=1 SV=2
MLLRSKPALPPPLMLLLLGPLGPLSPGALPRPAQAQDVVDLDFFT
QEPLHLVSPSFLSVTIDANLATDPRFLILLGSPKLRTLARGLSPA
YLRFGGTKTDFLIFDPKKESTFEERSYWQSQVNQDICKYGSIPPD
VEEKLRLEWPYQEQLLLREHYQKKFKNSTYSRSSVDVLYTFANCS
GLDLIFGLNALLRTADLQWNSSNAQLLLDYCSSKGYNISWELGNE
PNSFLKKADIFINGSQLGEDFIQLHKLLRKSTFKNAKLYGPDVGQ
PRRKTAKMLKSFLKAGGEVIDSVTWHHYYLNGRTATKEDFLNPDV
LDIFISSVQKVFQVVESTRPGKKVWLGETSSAYGGGAPLLSDTFA
AGFMWLDKLGLSARMGIEVVMRQVFFGAGNYHLVDENFDPLPDYW
LSLLFKKLVGTKVLMASVQGSKRRKLRVYLHCTNTDNPRYKEGDL
TLYAINLHNVTKYLRLPYPFSNKQVDKYLLRPLGPHGLLSKSVQL
NGLTLKMVDDQTLPPLMEKPLRPGSSLGLPAFSYSFFVIRNAKVA
ACI
"""

VERSICAN = """
>sp|P13611|CSPG2_HUMAN Versican core protein OS=Homo sapiens GN=VCAN PE=1 SV=3
MFINIKSILWMCSTLIVTHALHKVKVGKSPPVRGSLSGKVSLPCHFSTMPTLPPSYNTSEFLRIKWSKIEVDKNGKDLKETTVLVAQNGN
IKIGQDYKGRVSVPTHPEAVGDASLTVVKLLASDAGLYRCDVMYGIEDTQDTVSLTVDGVVFHYRAATSRYTLNFEAAQKACLDVGAVIA
TPEQLFAAYEDGFEQCDAGWLADQTVRYPIRAPRVGCYGDKMGKAGVRTYGFRSPQETYDVYCYVDHLDGDVFHLTVPSKFTFEEAAKEC
ENQDARLATVGELQAAWRNGFDQCDYGWLSDASVRHPVTVARAQCGGGLLGVRTLYRFENQTGFPPPDSRFDAYCFKPKEATTIDLSILA
ETASPSLSKEPQMVSDRTTPIIPLVDELPVIPTEFPPVGNIVSFEQKATVQPQAITDSLATKLPTPTGSTKKPWDMDDYSPSASGPLGKL
DISEIKEEVLQSTTGVSHYATDSWDGVVEDKQTQESVTQIEQIEVGPLVTSMEILKHIPSKEFPVTETPLVTARMILESKTEKKMVSTVS
ELVTTGHYGFTLGEEDDEDRTLTVGSDESTLIFDQIPEVITVSKTSEDTIHTHLEDLESVSASTTVSPLIMPDNNGSSMDDWEERQTSGR
ITEEFLGKYLSTTPFPSQHRTEIELFPYSGDKILVEGISTVIYPSLQTEMTHRRERTETLIPEMRTDTYTDEIQEEITKSPFMGKTEEEV
FSGMKLSTSLSEPIHVTESSVEMTKSFDFPTLITKLSAEPTEVRDMEEDFTATPGTTKYDENITTVLLAHGTLSVEAATVSKWSWDEDNT
TSKPLESTEPSASSKLPPALLTTVGMNGKDKDIPSFTEDGADEFTLIPDSTQKQLEEVTDEDIAAHGKFTIRFQPTTSTGIAEKSTLRDS
TTEEKVPPITSTEGQVYATMEGSALGEVEDVDLSKPVSTVPQFAHTSEVEGLAFVSYSSTQEPTTYVDSSHTIPLSVIPKTDWGVLVPSV
PSEDEVLGEPSQDILVIDQTRLEATISPETMRTTKITEGTTQEEFPWKEQTAEKPVPALSSTAWTPKEAVTPLDEQEGDGSAYTVSEDEL
LTGSERVPVLETTPVGKIDHSVSYPPGAVTEHKVKTDEVVTLTPRIGPKVSLSPGPEQKYETEGSSTTGFTSSLSPFSTHITQLMEETTT
EKTSLEDIDLGSGLFEKPKATELIEFSTIKVTVPSDITTAFSSVDRLHTTSAFKPSSAITKKPPLIDREPGEETTSDMVIIGESTSHVPP
TTLEDIVAKETETDIDREYFTTSSPPATQPTRPPTVEDKEAFGPQALSTPQPPASTKFHPDINVYIIEVRENKTGRMSDLSVIGHPIDSE
SKEDEPCSEETDPVHDLMAEILPEFPDIIEIDLYHSEENEEEEEECANATDVTTTPSVQYINGKHLVTTVPKDPEAAEARRGQFESVAPS
QNFSDSSESDTHPFVIAKTELSTAVQPNESTETTESLEVTWKPETYPETSEHFSGGEPDVFPTVPFHEEFESGTAKKGAESVTERDTEVG
HQAHEHTEPVSLFPEESSGEIAIDQESQKIAFARATEVTFGEEVEKSTSVTYTPTIVPSSASAYVSEEEAVTLIGNPWPDDLLSTKESWV
EATPRQVVELSGSSSIPITEGSGEAEEDEDTMFTMVTDLSQRNTTDTLITLDTSRIITESFFEVPATTIYPVSEQPSAKVVPTKFVSETD
TSEWISSTTVEEKKRKEEEGTTGTASTFEVYSSTQRSDQLILPFELESPNVATSSDSGTRKSFMSLTTPTQSEREMTDSTPVFTETNTLE
NLGAQTTEHSSIHQPGVQEGLTTLPRSPASVFMEQGSGEAAADPETTTVSSFSLNVEYAIQAEKEVAGTLSPHVETTFSTEPTGLVLSTV
MDRVVAENITQTSREIVISERLGEPNYGAEIRGFSTGFPLEEDFSGDFREYSTVSHPIAKEETVMMEGSGDAAFRDTQTSPSTVPTSVHI
SHISDSEGPSSTMVSTSAFPWEEFTSSAEGSGEQLVTVSSSVVPVLPSAVQKFSGTASSIIDEGLGEVGTVNEIDRRSTILPTAEVEGTK
APVEKEEVKVSGTVSTNFPQTIEPAKLWSRQEVNPVRQEIESETTSEEQIQEEKSFESPQNSPATEQTIFDSQTFTETELKTTDYSVLTT
KKTYSDDKEMKEEDTSLVNMSTPDPDANGLESYTTLPEATEKSHFFLATALVTESIPAEHVVTDSPIKKEESTKHFPKGMRPTIQESDTE
LLFSGLGSGEEVLPTLPTESVNFTEVEQINNTLYPHTSQVESTSSDKIEDFNRMENVAKEVGPLVSQTDIFEGSGSVTSTTLIEILSDTG
AEGPTVAPLPFSTDIGHPQNQTVRWAEEIQTSRPQTITEQDSNKNSSTAEINETTTSSTDFLARAYGFEMAKEFVTSAPKPSDLYYEPSG
EGSGEVDIVDSFHTSATTQATRQESSTTFVSDGSLEKHPEVPSAKAVTADGFPTVSVMLPLHSEQNKSSPDPTSTLSNTVSYERSTDGSF
QDRFREFEDSTLKPNRKKPTENIIIDLDKEDKDLILTITESTILEILPELTSDKNTIIDIDHTKPVYEDILGMQTDIDTEVPSEPHDSND
ESNDDSTQVQEIYEAAVNLSLTEETFEGSADVLASYTQATHDESMTYEDRSQLDHMGFHFTTGIPAPSTETELDVLLPTATSLPIPRKSA
TVIPEIEGIKAEAKALDDMFESSTLSDGQAIADQSEIIPTLGQFERTQEEYEDKKHAGPSFQPEFSSGAEEALVDHTPYLSIATTHLMDQ
SVTEVPDVMEGSNPPYYTDTTLAVSTFAKLSSQTPSSPLTIYSGSEASGHTEIPQPSALPGIDVGSSVMSPQDSFKEIHVNIEATFKPSS
EEYLHITEPPSLSPDTKLEPSEDDGKPELLEEMEASPTELIAVEGTEILQDFQNKTDGQVSGEAIKMFPTIKTPEAGTVITTADEIELEG
ATQWPHSTSASATYGVEAGVVPWLSPQTSERPTLSSSPEINPETQAALIRGQDSTIAASEQQVAARILDSNDQATVNPVEFNTEVATPPF
SLLETSNETDFLIGINEESVEGTAIYLPGPDRCKMNPCLNGGTCYPTETSYVCTCVPGYSGDQCELDFDECHSNPCRNGATCVDGFNTFR
CLCLPSYVGALCEQDTETCDYGWHKFQGQCYKYFAHRRTWDAAERECRLQGAHLTSILSHEEQMFVNRVGHDYQWIGLNDKMFEHDFRWT
DGSTLQYENWRPNQPDSFFSAGEDCVVIIWHENGQWNDVPCNYHLTYTCKKGTVACGQPPVVENAKTFGKMKPRYEINSLIRYHCKDGFI
QRHLPTIRCLGNGRWAIPKITCMNPSAYQRTYSMKYFKNSSSAKDNSINTSKHDHRWSRRWQESRR
"""

o_glycans = """
{Hex:1; HexNAc:1; Neu5Ac:2}    O-Glycan
"""

decorin = """
>sp|P21793|PGS2_BOVIN Decorin
MKATIIFLLVAQVSWAGPFQQKGLFDFMLEDEASGIGPEEHFPEVPEIEPMGPVCPFRCQ
CHLRVVQCSDLGLEKVPKDLPPDTALLDLQNNKITEIKDGDFKNLKNLHTLILINNKISK
ISPGAFAPLVKLERLYLSKNQLKELPEKMPKTLQELRVHENEITKVRKSVFNGLNQMIVV
ELGTNPLKSSGIENGAFQGMKKLSYIRIADTNITTIPQGLPPSLTELHLDGNKITKVDAA
SLKGLNNLAKLGLSFNSISAVDNGSLANTPHLRELHLNNNKLVKVPGGLADHKYIQVVYL
HNNNISAIGSNDFCPPGYNTKKASYSGVSLFSNPVQYWEIQPSTFRCVYVRAAVQLGNYK
"""

constant_modifications = ["Carbamidomethyl (C)"]
variable_modifications = ["Deamidation (N)", "Pyro-glu from Q (Q@N-term)"]


mt = modification.RestrictedModificationTable(
    constant_modifications=constant_modifications,
    variable_modifications=variable_modifications)

variable_modifications = [mt[v] for v in variable_modifications]
constant_modifications = [mt[c] for c in constant_modifications]


class FastaGlycopeptideTests(unittest.TestCase):

    def setup_tempfile(self, source):
        file_name = tempfile.mktemp()
        open(file_name, 'w').write(source)
        return file_name

    def clear_file(self, path):
        open(path, 'wb')

    def test_build_hypothesis(self):
        glycan_file = self.setup_tempfile(GLYCAN_RULE_FILE_SOURCE)
        fasta_file = self.setup_tempfile(FASTA_FILE_SOURCE)
        db_file = fasta_file + '.db'

        glycan_builder = CombinatorialGlycanHypothesisSerializer(glycan_file, db_file)
        glycan_builder.start()

        glycopeptide_builder = naive_glycopeptide.MultipleProcessFastaGlycopeptideHypothesisSerializer(
            fasta_file, db_file, glycan_builder.hypothesis_id, constant_modifications=constant_modifications,
            variable_modifications=variable_modifications, max_missed_cleavages=1)
        glycopeptide_builder.start()

        self.assertEqual(231800, glycopeptide_builder.query(Glycopeptide).count())

        redundancy = glycopeptide_builder.query(
            Glycopeptide.glycopeptide_sequence,
            Protein.name,
            serialize.func.count(Glycopeptide.glycopeptide_sequence)).join(
            Glycopeptide.protein).join(Glycopeptide.peptide).group_by(
                Glycopeptide.glycopeptide_sequence,
                Protein.name,
                Peptide.start_position,
                Peptide.end_position).yield_per(1000)

        for sequence, protein, count in redundancy:
            self.assertEqual(count, 1, "%s in %s has multiplicity %d" % (sequence, protein, count))

        for case in glycopeptide_builder.query(Glycopeptide).filter(
                Glycopeptide.glycopeptide_sequence ==
                "SVQEIQATFFYFTPN(N-Glycosylation)K{Hex:5; HexNAc:4; Neu5Ac:2}").all():
            self.assertAlmostEqual(case.calculated_mass, 4123.718954557139, 5)

        self.clear_file(glycan_file)
        self.clear_file(fasta_file)
        self.clear_file(db_file)

    def test_missing_glycopeptide(self):
        glycan_file = self.setup_tempfile(o_glycans)
        fasta_file = self.setup_tempfile(HEPARANASE)
        db_file = fasta_file + '.db'

        glycan_builder = TextFileGlycanHypothesisSerializer(glycan_file, db_file)
        glycan_builder.start()

        glycopeptide_builder = naive_glycopeptide.FastaGlycopeptideHypothesisSerializer(
            fasta_file, db_file, glycan_builder.hypothesis_id, protease='trypsin',
            constant_modifications=constant_modifications,
            max_missed_cleavages=2)
        glycopeptide_builder.start()

        case = glycopeptide_builder.query(
            Glycopeptide.glycopeptide_sequence == "KFKNSTYS(O-Glycosylation)R{Hex:1; HexNAc:1; Neu5Ac:2}").first()
        self.assertIsNotNone(case)

        self.clear_file(glycan_file)
        self.clear_file(fasta_file)
        self.clear_file(db_file)

    def test_throughput(self):
        fasta_file = fixtures.get_test_data("phil-82-proteins.fasta")
        glycan_file = fixtures.get_test_data("IAV_matched_glycans.txt")
        db_file = self.setup_tempfile("")
        print(db_file)

        glycan_builder = TextFileGlycanHypothesisSerializer(glycan_file, db_file)
        glycan_builder.start()

        glycopeptide_builder = naive_glycopeptide.MultipleProcessFastaGlycopeptideHypothesisSerializer(
            fasta_file, db_file, glycan_builder.hypothesis_id, constant_modifications=constant_modifications,
            variable_modifications=variable_modifications, max_missed_cleavages=2)
        glycopeptide_builder.start()
        self.clear_file(db_file)

    def test_uniprot_info(self):
        fasta_file = self.setup_tempfile(decorin)
        glycan_file = self.setup_tempfile(o_glycans)
        db_file = self.setup_tempfile("")
        glycan_builder = TextFileGlycanHypothesisSerializer(glycan_file, db_file)
        glycan_builder.start()

        glycopeptide_builder = naive_glycopeptide.MultipleProcessFastaGlycopeptideHypothesisSerializer(
            fasta_file, db_file, glycan_builder.hypothesis_id,
            protease=['trypsin'],
            constant_modifications=constant_modifications,
            variable_modifications=[], max_missed_cleavages=2)
        glycopeptide_builder.start()

        post_cleavage = glycopeptide_builder.query(serialize.Peptide).filter(
            serialize.Peptide.base_peptide_sequence == "DEASGIGPEEHFPEVPEIEPMGPVCPFR").first()
        self.assertIsNotNone(post_cleavage)

        self.clear_file(db_file)
        self.clear_file(fasta_file)
        self.clear_file(glycan_file)


if __name__ == '__main__':
    unittest.main()
