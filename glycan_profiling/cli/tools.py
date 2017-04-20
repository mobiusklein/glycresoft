import sys
import cmd
import csv
import threading
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty
import click

from glycopeptidepy.io import fasta, uniprot
from glycopeptidepy.structure.residue import UnknownAminoAcidException

from .base import cli
from glycan_profiling.serialize import (
    DatabaseBoundOperation, GlycanHypothesis, GlycopeptideHypothesis,
    SampleRun, Analysis)

from glycan_profiling.database.builder.glycopeptide.proteomics import mzid_proteome


@cli.group("tools", short_help="Odds and ends to help inspect data and diagnose issues.")
def tools():
    pass


@tools.command("list", short_help='List names and ids of collections in the database')
@click.pass_context
@click.argument("database-connection", type=DatabaseBoundOperation)
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
    session = db.session()
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

    with open(path, 'r') as handle:
        invalid_sequences = []
        for entry in fasta.FastaFileParser(handle):
            try:
                fasta.ProteinSequence(entry['name'], entry['sequence'])
            except UnknownAminoAcidException as e:
                invalid_sequences.append((entry['name'], e))
    for name, error in invalid_sequences:
        click.secho("%s had %s" % (name, error), fg='yellow')


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
def known_uniprot_glycoprotein(file_path=None, fasta_format=False):
    if file_path is not None:
        handle = open(file_path)
    else:
        handle = sys.stdin

    if fasta_format:
        reader = fasta.ProteinFastaFileParser(handle)

        def name_getter(x):
            return x.name
    else:
        reader = handle

        def name_getter(x):
            return fasta.default_parser(x)

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
                if has_known_glycosylation(name_getter(protein).accession):
                    outqueue.put(protein)
            except Exception as e:
                print(e, protein, type(protein), protein)

    def consumer_task(outqueue, no_more_event):
        has_work = True
        if fasta_format:
            writer = fasta.ProteinFastaFileWriter(sys.stdout)
            write_fn = writer.write
        else:
            def write_fn(payload):
                sys.stdout.write(str(payload).strip() + '\n')
        while has_work:
            try:
                protein = outqueue.get(True, 1)
            except Empty:
                if no_more_event.is_set():
                    has_work = False
                continue
            write_fn(protein)
        sys.stdout.close()

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
