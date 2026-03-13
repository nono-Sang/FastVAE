from setuptools import find_namespace_packages, setup


setup(
    name="fastvae",
    version="0.0.0",
    description="FastVAE",
    packages=find_namespace_packages(include=["fastvae*", "test*"]),
    include_package_data=True,
    python_requires=">=3.8",
    license="Apache-2.0",
)
