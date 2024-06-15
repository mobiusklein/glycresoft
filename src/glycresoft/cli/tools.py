import sys
import cmd
import csv
import threading
import code
import pprint

from queue import Queue, Empty

import click
import pkg_resources

from glypy.structure.glycan_composition import HashableGlycanComposition
from glycopeptidepy.io import fasta, uniprot
from glycopeptidepy.structure.residue import UnknownAminoAcidException

from .base import cli
from .validators import RelativeMassErrorParam, get_by_name_or_id

from glycresoft.serialize import (
    DatabaseBoundOperation, GlycanHypothesis, GlycopeptideHypothesis,
    SampleRun, Analysis, Protein, Peptide, Glycopeptide, GlycanClass,
    GlycanComposition, func, FileBlob)

from glycresoft.database import (
    GlycopeptideDiskBackedStructureDatabase,
    GlycanCompositionDiskBackedStructureDatabase)

from glycresoft.database.builder.glycopeptide.proteomics import mzid_proteome
from glycresoft.database.builder.glycopeptide.proteomics.uniprot import UniprotProteinDownloader


@cli.group("tools", short_help="Odds and ends to help inspect data and diagnose issues")
def tools():
    pass


def database_connection(fn):
    arg = click.argument("database-connection", doc_help=(
        "A connection URI for a database, or a path on the file system"),
        type=DatabaseBoundOperation)
    return arg(fn)


def hypothesis_identifier(hypothesis_type):
    def wrapper(fn):
        arg = click.argument("hypothesis-identifier", doc_help=(
            "The ID number or name of the %s hypothesis to use" % (hypothesis_type,)))
        return arg(fn)
    return wrapper


def analysis_identifier(fn):
    arg = click.argument("analysis-identifier", doc_help=(
        "The ID number or name of the analysis to use"))
    return arg(fn)


@tools.command("list", short_help='List names and ids of collections in the database')
@click.pass_context
@database_connection
def list_contents(context, database_connection):
    click.echo("Glycan Hypothesis")
    for hypothesis in database_connection.query(GlycanHypothesis):
        click.echo("\t%d\t%s\t%s" % (hypothesis.id, hypothesis.name, hypothesis.uuid))

    click.echo("\nGlycopeptide Hypothesis")
    for hypothesis in database_connection.query(GlycopeptideHypothesis):
        click.echo("\t%d\t%s\t%s\t%d" % (hypothesis.id, hypothesis.name, hypothesis.uuid,
                                         hypothesis.glycan_hypothesis.id))

    click.echo("\nSample Run")
    for run in database_connection.query(SampleRun):
        click.echo("\t%d\t%s\t%s" % (run.id, run.name, run.uuid))

    click.echo("\nAnalysis")
    for analysis in database_connection.query(Analysis):
        click.echo("\t%d\t%s\t%s" % (analysis.id, analysis.name, analysis.sample_run.name))

    click.echo("\nFile Blobs")
    for blob in database_connection.query(FileBlob):
        click.echo("\t%d\t%s" % (blob.id, blob.name))


@tools.command('mzid-proteins', short_help='Extract proteins from mzIdentML files')
@click.argument("mzid-file")
def list_protein_names(mzid_file):
    for name in mzid_proteome.protein_names(mzid_file):
        click.echo(name)


class SQLShellInterpreter(cmd.Cmd):
    prompt = '[sql-shell] '

    def __init__(self, session, *args, **kwargs):
        self.session = session
        cmd.Cmd.__init__(self, *args, **kwargs)

    def postcmd(self, stop, line):
        if line == "quit":
            return True
        return False

    def precmd(self, line):
        tokens = line.split(" ")
        tokens[0] = tokens[0].lower()
        return ' '.join(tokens)

    def _print_table(self, result):
        rows = list(result)
        if len(rows) == 0:
            return
        sizes = [0] * len(rows[0])
        titles = rows[0].keys()
        str_rows = []
        for row in rows:
            str_row = []
            for i, col in enumerate(row):
                col = repr(col)
                col_len = len(col)
                if sizes[i] < col_len:
                    sizes[i] = col_len
                str_row.append(col)
            str_rows.append(str_row)
        print(" | ".join(titles))
        for row in str_rows:
            print(' | '.join(row))

    def _to_csv(self, rows, fh):
        titles = rows[0].keys()
        writer = csv.writer(fh)
        writer.writerow(titles)
        writer.writerows(rows)

    def do_export(self, line):
        try:
            fname, line = line.rsplit(";", 1)
            if len(fname.strip()) == 0 or len(line.strip()) == 0:
                raise ValueError()
            try:
                result = self.session.execute("select " + line)
            except Exception as e:
                print(str(e))
                self.session.rollback()
                return
            try:
                rows = list(result)
                print("%d rows selected; Writing to %r" % (len(rows), fname))
                with open(fname, 'wb') as handle:
                    self._to_csv(rows, handle)
            except Exception as e:
                print(str(e))

        except ValueError:
            print("Could not determine the file name to export to.")
            print("Usage: export <filename>; <query>")

    def do_execsql(self, line):
        try:
            result = self.session.execute(line)
            self._print_table(result)

        except Exception as e:
            print(str(e))
            self.session.rollback()
        return False

    def do_explain(self, line):
        try:
            result = self.session.execute("explain " + line)
            self._print_table(result)

        except Exception as e:
            print(str(e))
            self.session.rollback()
        return False

    def do_select(self, line):
        try:
            result = self.session.execute("select " + line)
            self._print_table(result)

        except Exception as e:
            print(str(e))
            self.session.rollback()
        return False

    def do_quit(self, line):
        return True


@tools.command("sql-shell",
               short_help=("A minimal SQL command shell for running "
                           "diagnostics and exporting arbitrary data."))
@click.argument("database-connection")
@click.option("-s", "--script", default=None)
def sql_shell(database_connection, script=None):
    db = DatabaseBoundOperation(database_connection)
    session = db.session()  # pylint: disable=not-callable
    interpreter = SQLShellInterpreter(session)
    if script is None:
        interpreter.cmdloop()
    else:
        result = session.execute(script)
        interpreter._to_csv(list(result), sys.stdout)


@tools.command("validate-fasta", short_help="Validates a FASTA file, checking a few errors.")
@click.argument("path")
def validate_fasta(path):
    with open(path, "r") as handle:
        n_deflines = 0
        for line in handle:
            if line.startswith(">"):
                n_deflines += 1
    with open(path, 'r') as handle:
        n_entries = 0
        for entry in fasta.FastaFileParser(handle):
            n_entries += 1
    if n_entries != n_deflines:
        click.secho("%d\">\" prefixed lines found, but %d entries parsed" % (n_deflines, n_entries))
    else:
        click.echo("%d Protein Sequences" % (n_entries, ))
    n_glycoprots = 0
    o_glycoprots = 0
    with open(path, 'r') as handle:
        invalid_sequences = []
        for entry in fasta.FastaFileParser(handle):
            try:
                seq = fasta.ProteinSequence(entry['name'], entry['sequence'])
                n_glycoprots += bool(seq.n_glycan_sequon_sites)
                o_glycoprots += bool(seq.o_glycan_sequon_sites)
            except UnknownAminoAcidException as e:
                invalid_sequences.append((entry['name'], e))
    click.echo("Proteins with N-Glycosites: %d" % n_glycoprots)
    for name, error in invalid_sequences:
        click.secho("%s had %s" % (name, error), fg='yellow')


@tools.command("validate-glycan-text", short_help="Validates a text file of glycan compositions")
@click.argument("path")
def validate_glycan_text(path):
    from glycresoft.database.builder.glycan.glycan_source import TextFileGlycanCompositionLoader
    with open(path, 'r') as handle:
        loader = TextFileGlycanCompositionLoader(handle)
        n = 0
        glycan_classes = set()
        residues = set()
        unresolved = set()
        for line in loader:
            n += 1
            glycan_classes.update(line[1])
            glycan_composition = HashableGlycanComposition.parse(line[0])
            for residue in glycan_composition.keys():
                if residue.mass() == 0:
                    unresolved.add(residue)
                residues.add(residue)
        click.secho("%d glycan compositions" % (n,))
        click.secho("Residues:")
        for residue in residues:
            click.secho("\t%s - %f" % (str(residue), residue.mass()))
        if unresolved:
            click.secho("Unresolved Residues:", fg='yellow')
            click.secho("\n".join(str(r) for r in unresolved), fg='yellow')


def has_known_glycosylation(accession):
    try:
        prot = uniprot.get(accession)
        if "Glycoprotein" in prot.keywords:
            return True
        else:
            # for feature in prot.features:
            #     if isinstance(feature, uniprot.GlycosylationSite):
            #         return True
            pass
        return False
    except Exception:
        return False


@tools.command("known-glycoproteins", short_help=(
    'Checks UniProt to see if a list of proteins contains any known glycoproteins'))
@click.option("-i", "--file-path", help="Read input from a file instead of STDIN")
@click.option("-f", "--fasta-format", is_flag=True, help="Indicate input is in FASTA format")
@click.option("-o", "--output-path", help="Write output to a file instead of STDOUT")
def known_uniprot_glycoprotein(file_path=None, output_path=None, fasta_format=False):
    if file_path is not None:
        handle = open(file_path)
    else:
        handle = sys.stdin

    if fasta_format:
        reader = fasta.ProteinFastaFileParser(handle)

        def name_getter(x):
            return x.name.accession
    else:
        reader = handle

        def name_getter(x):
            header = fasta.default_parser(x)
            try:
                return header.accession
            except Exception:
                return header[0]

    if output_path is None:
        outhandle = sys.stdout
    else:
        outhandle = open(output_path, 'w')

    def checker_task(inqueue, outqueue, no_more_event):
        has_work = True
        while has_work:
            try:
                protein = inqueue.get(True, 1)
            except Empty:
                if no_more_event.is_set():
                    has_work = False
                continue
            try:
                if has_known_glycosylation(name_getter(protein)):
                    outqueue.put(protein)
            except Exception as e:
                click.secho("%r occurred for %s" % (e, protein), err=True, fg='yellow')

    def consumer_task(outqueue, no_more_event):
        has_work = True
        if fasta_format:
            writer = fasta.ProteinFastaFileWriter(outhandle)
            write_fn = writer.write
        else:
            def write_fn(payload):
                outhandle.write(str(payload).strip() + '\n')
        while has_work:
            try:
                protein = outqueue.get(True, 1)
            except Empty:
                if no_more_event.is_set():
                    has_work = False
                continue
            write_fn(protein)
        outhandle.close()

    producer_done = threading.Event()
    checker_done = threading.Event()

    inqueue = Queue()
    outqueue = Queue()

    n_threads = 10
    checkers = [threading.Thread(
        target=checker_task, args=(inqueue, outqueue, producer_done)) for i in range(n_threads)]
    for check in checkers:
        check.start()

    consumer = threading.Thread(target=consumer_task, args=(outqueue, checker_done))
    consumer.start()

    for protein in reader:
        inqueue.put(protein)

    producer_done.set()

    for checker in checkers:
        checker.join()
    checker_done.set()
    consumer.join()


@tools.command("download-uniprot", short_help=(
    "Downloads a list of proteins from UniProt"))
@click.option("-i", "--name-file-path", help="Read input from a file instead of STDIN")
@click.option("-o", "--output-path", help="Write output to a file instead of STDOUT")
def download_uniprot(name_file_path=None, output_path=None):
    if name_file_path is not None:
        handle = open(name_file_path)
    else:
        handle = sys.stdin

    def name_getter(x):
        header = fasta.default_parser(x)
        try:
            return header.accession
        except Exception:
            return header[0]

    accession_list = []
    for line in handle:
        accession_list.append(name_getter(line))
    if output_path is None:
        outhandle = click.get_binary_stream('stdout')
    else:
        outhandle = open(output_path, 'wb')
    writer = fasta.FastaFileWriter(outhandle)
    downloader = UniprotProteinDownloader(accession_list, 10)
    downloader.start()
    has_work = True

    def make_protein(accession, uniprot_protein):
        sequence = uniprot_protein.sequence
        name = "sp|{accession}|{gene_name} {description}".format(
            accession=accession, gene_name=uniprot_protein.gene_name,
            description=uniprot_protein.recommended_name)
        return name, sequence

    while has_work:
        try:
            accession, value = downloader.get(True, 3)
            if isinstance(value, Exception):
                click.echo("Could not retrieve %s - %r" % (accession, value), err=True)
            else:
                writer.write(*make_protein(accession, value))

        except Empty:
            # If we haven't completed the download process, block
            # now and wait for the threads to join, then continue
            # trying to fetch results
            if not downloader.done_event.is_set():
                downloader.join()
                continue
            # Otherwise we've already waited for all the results to
            # arrive and we can stop iterating
            else:
                has_work = False


@tools.command("mass-search")
@click.option("-p", "--glycopeptide", is_flag=True)
@click.option("-m", "--error-tolerance", type=RelativeMassErrorParam(), default=1e-5)
@click.argument("database-connection")
@click.argument("hypothesis-identifier")
@click.argument("target-mass", type=float)
def mass_search(database_connection, hypothesis_identifier, target_mass, glycopeptide=False, error_tolerance=1e-5):
    if glycopeptide:
        handle = GlycopeptideDiskBackedStructureDatabase(database_connection, int(hypothesis_identifier))
    else:
        handle = GlycanCompositionDiskBackedStructureDatabase(database_connection, int(hypothesis_identifier))
    width = (target_mass * error_tolerance)
    click.secho("Mass Window: %f-%f" % (target_mass - width, target_mass + width), fg='yellow')
    hits = list(handle.search_mass_ppm(target_mass, error_tolerance))
    if not hits:
        click.secho("No Matches", fg='red')
    for hit in hits:
        click.echo("\t".join(map(str, hit)))


@tools.command("version-check")
def version_check():
    packages = [
        "glycresoft",
        "glycresoft_app",
        "glypy",
        "glycopeptidepy",
        "ms_peak_picker",
        "brain-isotopic-distribution",
        "ms_deisotope",
        "pyteomics",
        "lxml",
        "numpy",
        "scipy",
        "matplotlib"
    ]
    click.secho("Library Versions", fg='yellow')
    for dep in packages:
        try:
            rev = pkg_resources.require(dep)
            click.echo(str(rev[0]))
        except Exception:
            try:
                module = __import__(dep)
            except ImportError:
                continue
            version = getattr(module, "__version__", None)
            if version is None:
                version = getattr(module, "version", None)
            if version is None:
                try:
                    module = __import__("%s.version" % dep).version
                    version = module.version
                except ImportError:
                    continue
            if version:
                click.echo("%s %s" % (dep, version))


@tools.command("interactive-shell")
@click.option("-s", "--script", default=None)
@click.argument("script_args", nargs=-1)
def interactive_shell(script_args, script):
    if script:
        with open(script, 'rt') as fh:
            script = fh.read()
        exec(script)
    code.interact(local=locals())


@tools.command("update-analysis-parameters")
@database_connection
@analysis_identifier
@click.option("-p", "--parameter", nargs=2, multiple=True, required=False)
def update_analysis_parameters(database_connection, analysis_identifier, parameter):
    session = database_connection.session
    analysis = get_by_name_or_id(session, Analysis, analysis_identifier)
    click.echo("Current Parameters:")
    pprint.pprint(analysis.parameters)
    for name, value in parameter:
        analysis.parameters[name] = value
    session.add(analysis)
    session.commit()


@tools.command("summarize-glycopeptide-hypothesis",
               short_help="Show summary information about a glycopeptide hypothesis")
@database_connection
@hypothesis_identifier(GlycopeptideHypothesis)
def summarize_glycopeptide_hypothesis(database_connection, hypothesis_identifier):
    session = database_connection.session
    hypothesis = get_by_name_or_id(session, GlycopeptideHypothesis, hypothesis_identifier)
    gp_counts = session.query(Protein, func.count(Glycopeptide.id)).join(
        Glycopeptide).group_by(Protein.id).filter(
        Protein.hypothesis_id == hypothesis.id).order_by(
        Protein.id).all()
    pept_counts = session.query(Protein, func.count(Peptide.id)).join(
        Peptide).group_by(Protein.id).filter(
        Protein.hypothesis_id == hypothesis.id).order_by(
        Protein.id).all()

    gc_count = session.query(GlycanClass.name, func.count(GlycanClass.id)).join(
        GlycanComposition.structure_classes).filter(GlycanComposition.hypothesis_id == hypothesis.id).group_by(
            GlycanClass.id).order_by(GlycanClass.name).all()

    counts = {}
    for prot, c in gp_counts:
        counts[prot.id] = [prot, 0, c]
    for prot, c in pept_counts:
        try:
            counts[prot.id][1] = c
        except KeyError:
            counts[prot.id] = [prot, c, 0]

    counts = sorted(counts.values(), key=lambda x: x[::-1][:-1], reverse=1)
    pept_total = 0
    gp_total = 0
    for protein, pept_count, gp_count in counts:
        click.echo("%s: %d" % (protein.name, pept_count))
        pept_total += pept_count
        gp_total += gp_count

    click.echo("Total Peptides: %d" % (pept_total, ))
    click.echo("Total Glycopeptides: %d" % (gp_total, ))
    for cls_name, count in gc_count:
        click.echo("%s Glycan Compositions: %d" % (cls_name, count))
    click.echo(pprint.pformat(hypothesis.parameters))


@tools.command("extract-blob")
@database_connection
@click.argument("blob-identifier")
@click.option("-o", "--output-path", type=click.File(mode='wb'))
def extract_file_blob(database_connection, blob_identifier, output_path=None):
    if output_path is None:
        output_path = click.get_binary_stream('stdout')
    session = database_connection.session
    blob = get_by_name_or_id(session, FileBlob, blob_identifier)
    with blob.open() as fh:
        chunk_size = 2 ** 16
        chunk = fh.read(chunk_size)
        while chunk:
            output_path.write(chunk)
            chunk = fh.read(chunk_size)


@tools.command("csv-concat")
@click.argument("csv-paths", type=click.File(mode='rt'), nargs=-1)
@click.option("-o", "--output-path", type=click.Path(writable=True), help="Path to write output to")
def csv_concat(csv_paths, output_path=None):
    if output_path is None:
        output_path = '-'
    import csv
    headers = None
    with click.open_file(output_path, mode='wt') as outfh:
        writer = csv.writer(outfh)
        for infh in csv_paths:
            reader = csv.reader(infh)
            _header_line = next(reader)
            if headers is None:
                headers = _header_line
                writer.writerow(headers)
            elif _header_line != headers:
                raise click.ClickException("File %s did not have matching headers (%s != %s)" % (
                    infh.name, _header_line, headers))
            for row in reader:
                writer.writerow(row)
            infh.close()
            outfh.flush()
