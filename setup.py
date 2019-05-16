import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyCOMPASS",
    version="0.0.6",
    author="Marco Moretto",
    author_email="marco.moretto@fmach.it",
    description="A Python interface to COMPASS, the gene expression compendia GraphQL endpoint",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'requests==2.21.0',
        'numpy==1.16.0'
    ],
)