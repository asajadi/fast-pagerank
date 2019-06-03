import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fast-pagerank",
    version="0.0.3",
    author="Armin Sajadi",
    author_email="asajadi@gmail.com",
    description="A fast PageRank and Personalized PageRank implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/asajadi/fast_pagerank",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
