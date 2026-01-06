from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#") and not line.startswith("torch")]

setup(
    name="hyperresashs",
    version="1.0.0",
    author="HyperResASHS Team",
    description="Isotropic segmentation pipeline for MTL subregions from multi-modality 3T MRI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liyue3780/HyperResASHS",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)

