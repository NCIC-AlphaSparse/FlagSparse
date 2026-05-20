from setuptools import find_packages, setup

setup(
    name="flagsparse",
    version="1.0.0",
    description="FlagSparse - GPU sparse operations package",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.12",
    install_requires=[],
    include_package_data=True,
)
