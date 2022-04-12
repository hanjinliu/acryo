from setuptools import setup, find_packages

with open("tomtools/__init__.py", encoding="utf-8") as f:
    line = next(f)
    VERSION = line.strip().split()[-1][1:-1]

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="tomtools",
    version=VERSION,
    description="An extensible subtomogram averaging toolkit for Python.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Hanjin Liu",
    author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
    license="BSD 3-Clause",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "impy-array>=2.0.0",
    ],
    python_requires=">=3.8",
)
