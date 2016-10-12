from setuptools import setup, find_packages


def run_setup(include_cext=True):
    setup(
        name='glycan_profiling',
        version='0.0.1',
        packages=find_packages(),
        author=', '.join(["Joshua Klein"]),
        author_email=["jaklein@bu.edu"],
        entry_points={
            'console_scripts': [
                "glycresoft = glycan_profiling.cli.__main__:main"
            ],
        },
        classifiers=[
                'Development Status :: 3 - Alpha',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: BSD License',
                'Topic :: Scientific/Engineering :: Bio-Informatics'])


run_setup()
