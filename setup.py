# setup.py
"""
Setup script for semantic video colorization project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="semantic-video-colorization",
    version="1.0.0",
    author="Sita Ganesh",
    author_email="sitaganesh07@gmail.com",
    description="Real-time semantic video colorization using PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SitaGanesh/semantic-video-colorization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "colorize-video=src.inference:main",
            "train-colorizer=src.train:main",
        ],
    },
)
