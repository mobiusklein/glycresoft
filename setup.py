import sys
import traceback
import os
from setuptools import setup, Extension, find_packages

from distutils.command.build_ext import build_ext
from distutils.errors import (CCompilerError, DistutilsExecError,
                              DistutilsPlatformError)


def has_option(name):
    try:
        sys.argv.remove('--%s' % name)
        return True
    except ValueError:
        pass
    # allow passing all cmd line options also as environment variables
    env_val = os.getenv(name.upper().replace('-', '_'), 'false').lower()
    if env_val == "true":
        return True
    return False


include_diagnostics = has_option("include-diagnostics")
force_cythonize = has_option("force-cythonize")


def make_extensions():
    is_ci = bool(os.getenv("CI", ""))
    try:
        import numpy
    except ImportError:
        print("Installation requires `numpy`")
        raise
    macros = []
    try:
        from Cython.Build import cythonize
        cython_directives = {
            'embedsignature': True,
            "profile": include_diagnostics
        }
        if include_diagnostics:
            macros.append(("CYTHON_TRACE_NOGIL", "1"))
        if is_ci and include_diagnostics:
            cython_directives['linetrace'] = True
        extensions = cythonize(
            [
                Extension(
                    name="glycresoft._c.structure.fragment_match_map",
                    sources=["src/glycresoft/_c/structure/fragment_match_map.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.structure.intervals",
                    sources=["src/glycresoft/_c/structure/intervals.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.scoring.shape_fitter",
                    sources=["src/glycresoft/_c/scoring/shape_fitter.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.chromatogram_tree.mass_shift",
                    sources=["src/glycresoft/_c/chromatogram_tree/mass_shift.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.chromatogram_tree.index",
                    sources=["src/glycresoft/_c/chromatogram_tree/index.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.tandem.core_search",
                    sources=["src/glycresoft/_c/tandem/core_search.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.database.mass_collection",
                    sources=["src/glycresoft/_c/database/mass_collection.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.tandem.tandem_scoring_helpers",
                    libraries=["npymath"],
                    library_dirs=[os.path.join(os.path.dirname(numpy.get_include()), "lib")],
                    sources=["src/glycresoft/_c/tandem/tandem_scoring_helpers.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.tandem.spectrum_match",
                    sources=["src/glycresoft/_c/tandem/spectrum_match.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.composition_network.graph",
                    sources=["src/glycresoft/_c/composition_network/graph.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.composition_distribution_model.utils",
                    sources=["src/glycresoft/_c/composition_distribution_model/utils.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.structure.lru", sources=["src/glycresoft/_c/structure/lru.pyx"]
                ),
                Extension(
                    name="glycresoft._c.tandem.target_decoy",
                    sources=["src/glycresoft/_c/tandem/target_decoy.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.structure.structure_loader",
                    sources=["src/glycresoft/_c/structure/structure_loader.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.tandem.oxonium_ions",
                    sources=["src/glycresoft/_c/tandem/oxonium_ions.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.structure.probability",
                    sources=["src/glycresoft/_c/structure/probability.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.tandem.peptide_graph",
                    sources=["src/glycresoft/_c/tandem/peptide_graph.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
                Extension(
                    name="glycresoft._c.scoring.elution_time_grouping",
                    sources=["src/glycresoft/_c/scoring/elution_time_grouping.pyx"],
                    include_dirs=[numpy.get_include()],
                ),
            ],
            compiler_directives=cython_directives,
            force=force_cythonize,
        )
    except ImportError as err:
        print(err)
        extensions = [
            Extension(
                name="glycresoft._c.structure.fragment_match_map",
                sources=["src/glycresoft/_c/structure/fragment_match_map.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.structure.intervals",
                sources=["src/glycresoft/_c/structure/intervals.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.scoring.shape_fitter",
                sources=["src/glycresoft/_c/scoring/shape_fitter.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.chromatogram_tree.mass_shift",
                sources=["src/glycresoft/_c/chromatogram_tree/mass_shift.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.chromatogram_tree.index",
                sources=["src/glycresoft/_c/chromatogram_tree/index.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.tandem.core_search",
                sources=["src/glycresoft/_c/tandem/core_search.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.database.mass_collection",
                sources=["src/glycresoft/_c/database/mass_collection.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.tandem.tandem_scoring_helpers",
                libraries=["npymath"],
                library_dirs=[os.path.join(os.path.dirname(numpy.get_include()), "lib")],
                sources=["src/glycresoft/_c/tandem/tandem_scoring_helpers.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.tandem.spectrum_match",
                sources=["src/glycresoft/_c/tandem/spectrum_match.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.composition_network.graph",
                sources=["src/glycresoft/_c/composition_network/graph.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.composition_distribution_model.utils",
                sources=["src/glycresoft/_c/composition_distribution_model/utils.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(name="glycresoft._c.structure.lru", sources=["src/glycresoft/_c/structure/lru.c"]),
            Extension(
                name="glycresoft._c.tandem.target_decoy",
                sources=["src/glycresoft/_c/tandem/target_decoy.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.structure.structure_loader",
                sources=["src/glycresoft/_c/structure/structure_loader.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.tandem.oxonium_ions",
                sources=["src/glycresoft/_c/tandem/oxonium_ions.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.structure.probability",
                sources=["src/glycresoft/_c/structure/probability.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.tandem.peptide_graph",
                sources=["src/glycresoft/_c/tandem/peptide_graph.c"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                name="glycresoft._c.scoring.elution_time_grouping",
                sources=["src/glycresoft/_c/scoring/elution_time_grouping.c"],
                include_dirs=[numpy.get_include()],
            ),
        ]
    return extensions


ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)
if sys.platform == 'win32':
    # 2.6's distutils.msvc9compiler can raise an IOError when failing to
    # find the compiler
    ext_errors += (IOError,)


class BuildFailed(Exception):

    def __init__(self):
        self.cause = sys.exc_info()[1]  # work around py 2/3 different syntax

    def __str__(self):
        return str(self.cause)


class ve_build_ext(build_ext):
    # This class allows C extension building to fail.

    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError:
            traceback.print_exc()
            raise BuildFailed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except ext_errors:
            traceback.print_exc()
            raise BuildFailed()
        except ValueError:
            # this can happen on Windows 64 bit, see Python issue 7511
            traceback.print_exc()
            if "'path'" in str(sys.exc_info()[1]):  # works with both py 2/3
                raise BuildFailed()
            raise


cmdclass = {}

cmdclass['build_ext'] = ve_build_ext


def status_msgs(*msgs):
    print('*' * 75)
    for msg in msgs:
        print(msg)
    print('*' * 75)


with open("src/glycresoft/version.py") as version_file:
    version = None
    for line in version_file.readlines():
        if "version = " in line:
            version = line.split(" = ")[1].replace("\"", "").strip()
            print("Version is: %r" % (version,))
            break
    else:
        print("Cannot determine version")


requirements = []
with open("requirements.txt") as requirements_file:
    requirements.extend(requirements_file.readlines())

try:
    with open("README.md") as readme_file:
        long_description = readme_file.read()
except Exception as e:
    print(e)
    long_description = ""


def run_setup(include_cext=True):

    setup(
        name="glycresoft",
        version=version,
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        include_package_data=True,
        author=", ".join(["Joshua Klein"]),
        author_email="jaklein@bu.edu",
        description="Glycan and Glycopeptide Mass Spectrometry Database Search Tool",
        long_description=long_description,
        long_description_content_type='text/markdown',
        entry_points={
            "console_scripts": ["glycresoft = glycresoft.cli.__main__:main"],
        },
        package_data={
            "glycresoft.models": ["src/glycresoft/models/data/*"],
            "glycresoft.database.prebuilt": ["src/glycresoft/database/prebuilt/data/*"],
            "glycresoft.output.report.glycan_lcms": ["src/glycresoft/output/report/glycan_lcms/*"],
            "glycresoft.output.report.glycopeptide_lcmsms": ["src/glycresoft/output/report/glycopeptide_lcmsms/*"],
        },
        ext_modules=make_extensions() if include_cext else None,
        cmdclass=cmdclass,
        install_requires=requirements,
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
        ],
        zip_safe=False,
        python_requires=">3.8",
        project_urls={
            "Documentation": "https://mobiusklein.github.io/glycresoft",
            "Source Code": "https://github.com/mobiusklein/glycresoft",
            "Issue Tracker": "https://github.com/mobiusklein/glycresoft/issues",
        },
    )


try:
    run_setup(True)
except Exception as exc:
    print(exc)
    run_setup(False)

    status_msgs(
        "WARNING: The C extension could not be compiled, " +
        "speedups are not enabled.",
        "Plain-Python build succeeded."
    )
