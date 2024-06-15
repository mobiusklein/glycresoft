import unittest
import tempfile
import warnings

from glycresoft.serialize.hypothesis.peptide import Peptide, Protein, Glycopeptide
from glycresoft.database.builder.glycopeptide import naive_glycopeptide
from glycresoft.database.builder.glycan import (
    TextFileGlycanHypothesisSerializer,
    CombinatorialGlycanHypothesisSerializer)
from glycresoft import serialize
from . import fixtures

from .test_constrained_combinatorics import FILE_SOURCE as GLYCAN_RULE_FILE_SOURCE

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

simple_n_glycans = """
{Hex:5; HexNAc:2}    N-Glycan
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
        file_name = tempfile.mktemp() + '.tmp'
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
            fasta_file, db_file, glycan_builder.hypothesis_id,
            use_uniprot=True,
            constant_modifications=constant_modifications,
            variable_modifications=variable_modifications,
            max_missed_cleavages=1)
        glycopeptide_builder.start()

        without_uniprot = 201400
        with_uniprot_without_variable_signal_peptide = 231800
        with_uniprot = 353400

        observed_count = glycopeptide_builder.query(Glycopeptide).count()

        assert observed_count in (
            with_uniprot, with_uniprot_without_variable_signal_peptide, without_uniprot)

        if observed_count == without_uniprot:
            warnings.warn("UniProt Annotations Not Used")
        if observed_count == with_uniprot_without_variable_signal_peptide:
            warnings.warn("Variable Signal Peptide Was Not Used")

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

        peptides_without_uniprot = 80
        peptides_with_uniprot = 88
        peptides_with_uniprot_and_ragged_signal_peptide = 140

        peptides_count = glycopeptide_builder.query(serialize.Peptide).count()

        if peptides_count == peptides_with_uniprot or peptides_count == peptides_with_uniprot_and_ragged_signal_peptide:
            post_cleavage = glycopeptide_builder.query(serialize.Peptide).filter(
                serialize.Peptide.base_peptide_sequence == "DEASGIGPEEHFPEVPEIEPMGPVCPFR").first()
            self.assertIsNotNone(post_cleavage)
            self.assertEqual(
                len(glycopeptide_builder.query(serialize.Protein).first().annotations), 2
            )
        elif peptides_count == peptides_without_uniprot:
            warnings.warn("Failed to communicate with UniProt, skip this test")
        else:
            raise ValueError("Incorrect peptide count: %r" % (peptides_count, ))

        self.clear_file(db_file)
        self.clear_file(fasta_file)
        self.clear_file(glycan_file)

    def test_extract_forward_backward(self):
        fasta_file = fixtures.get_test_data("yeast_glycoproteins.fa")
        glycan_file = self.setup_tempfile(simple_n_glycans)
        forward_db = self.setup_tempfile("")
        reverse_db = self.setup_tempfile("")

        glycan_builder = TextFileGlycanHypothesisSerializer(glycan_file, forward_db)
        glycan_builder.start()

        builder = naive_glycopeptide.MultipleProcessFastaGlycopeptideHypothesisSerializer(
            fasta_file, forward_db, 1)
        cnt = builder.extract_proteins()
        assert cnt == 251

        glycan_builder = TextFileGlycanHypothesisSerializer(glycan_file, reverse_db)
        glycan_builder.start()

        rev_builder = naive_glycopeptide.ReversingMultipleProcessFastaGlycopeptideHypothesisSerializer(
            fasta_file, reverse_db, 1)
        cnt = rev_builder.extract_proteins()
        assert cnt == 251
        fwd_prots = builder.query(serialize.Protein).all()
        rev_prots = rev_builder.query(serialize.Protein).all()

        for fx, rx in zip(fwd_prots, rev_prots):
            assert fx.name == rx.name
            assert len(fx.glycosylation_sites) == len(rx.glycosylation_sites)


if __name__ == '__main__':
    unittest.main()
