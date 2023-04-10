from setuptools import setup, find_packages

ACRYO = "acryo"

with open(f"{ACRYO}/__init__.py", encoding="utf-8") as f:
    line = next(f)
    VERSION = line.strip().split()[-1][1:-1]

with open("README.md", "r") as f:
    README = f.read()

setup(
    name=ACRYO,
    version=VERSION,
    description="An extensible cryo-EM/ET toolkit for Python.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Hanjin Liu",
    author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
    license="BSD 3-Clause",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7.3",
        "polars>=0.16.18",
        "dask>=2021.6.0",
        "typing_extensions>=4.1.1",
    ],
    python_requires=">=3.8",
)
