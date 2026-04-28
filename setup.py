from setuptools import setup, find_packages

setup(
    name="hanomi-feature-recognition",
    version="0.1.0",
    description="Metric learning on B-Rep subgraphs for CAD feature recognition",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
