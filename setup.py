import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="cs-vqe-test",
    use_scm_version=True,
    author="Tim Weaving",
    author_email="timothy.weaving.20@ucl.ac.uk",
    description="CS-VQE implementation with Orquestra.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TimWeaving/CS-VQE",
    packages=setuptools.find_namespace_packages(
        include=["utils", "cs_vqe_classes", "steps", "openfermionpyscf"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"],
    setup_requires=["setuptools_scm~=6.0"],
    dependency_links=['https://github.com/quantumlib/OpenFermion-PySCF/tarball/master#egg=openfermionpyscf-1.0']
)
