from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pricelens",
    version="0.1.0",
    author="PriceLens Team",
    author_email="",
    description="Real-time Pokemon card detection and price overlay system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PriceLens",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pricelens=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pricelens": ["config.yaml"],
    },
)