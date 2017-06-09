from setuptools import setup, find_packages

with open("glycan_profiling/version.py") as version_file:
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


def run_setup(include_cext=True):
    setup(
        name='glycan_profiling',
        version=version,
        packages=find_packages(),
        include_package_data=True,
        author=', '.join(["Joshua Klein"]),
        author_email=["jaklein@bu.edu"],
        entry_points={
            'console_scripts': [
                "glycresoft = glycan_profiling.cli.__main__:main"
            ],
        },
        package_data={
            "glycan_profiling.models": ["glycan_profiling/models/data/*"],
        },
        install_requires=requirements,
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Topic :: Scientific/Engineering :: Bio-Informatics'])


run_setup()
