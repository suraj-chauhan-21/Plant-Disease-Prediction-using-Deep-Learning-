from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="plant_disease_detection",
    version="1.0.0",
    author="Suraj Chauhan",
    description="A high-performance plant disease detection system using deep learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/animesh1012/machineLearning",
    packages=find_packages(include=["src", "src.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
