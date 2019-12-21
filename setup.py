from __future__ import absolute_import, division, print_function

import setuptools

setuptools.setup(
    name="iota",
    description="IOTA: Integration Optimization, Triage and Analysis",
    long_description="",
    long_description_content_type="text/x-rst",
    author="Leland Stanford Junior University",
    author_email="scientificsoftware@diamond.ac.uk",
    version="0.1",
    url="https://github.com/ssrl-px/iota",
    download_url="https://github.com/ssrl-px/iota/releases",
    license="BSD",
    install_requires=[],
    package_dir={'': 'src'},
    packages=["iota"],
    entry_points={
        "libtbx.dispatcher.script": [],
        "libtbx.precommit": ["iota = iota"],
    },
    scripts=[],
    tests_require=["mock", "procrunner", "pytest"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: POSIX :: Linux",
    ],
)
