import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="fastmri-recon",
    version="0.0.1",
    author="Zaccharie Ramzi",
    author_email="zaccharie.ramzi@gmail.com",
    description="Tools to benchmark different reconstruction neural nets on the fastMRI dataset",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/pypa/fastmri-reproducible-benchmark",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
