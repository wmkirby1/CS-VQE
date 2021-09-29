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
        include=["utils", "cs_vqe_classes", "hamiltonians", "steps", "scikit-quant"]),
    package_data={'utils': ['*.json']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"],
    setup_requires=["setuptools_scm~=6.0"],
    install_requires=['scikit-quant @ https://github.com/scikit-quant/scikit-quant/tarball/master#egg=scikit-quant-0.7.0']
)