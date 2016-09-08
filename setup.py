from setuptools import setup, find_packages


def run_setup(include_cext=True):
    setup(
        name='glycan_profiling',
        version='0.0.1',
        packages=find_packages("glycan_profiling"),
        author=', '.join(["Joshua Klein"]),
        author_email=["jaklein@bu.edu"],
        ext_modules=None,
        classifiers=[
                'Development Status :: 3 - Alpha',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: BSD License',
                'Topic :: Scientific/Engineering :: Bio-Informatics'])


run_setup()
