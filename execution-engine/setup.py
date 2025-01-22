# setup.py
from setuptools import setup, find_packages

setup(
    name="gpu-execution-engine",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "logging>=0.5.0"
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-asyncio',
            'pytest-cov',
            'black',
            'flake8',
        ]
    },
    python_requires='>=3.8',
)