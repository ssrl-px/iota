from __future__ import absolute_import, division, print_function

import setuptools

setuptools.setup(
    name="iota",
    description="IOTA: Integration Optimization, Triage and Analysis",
    long_description="",
    long_description_content_type="text/x-rst",
    author="Leland Stanford Junior University",
    author_email="scientificsoftware@diamond.ac.uk",
    version="1.5.4",
    url="https://github.com/ssrl-px/iota",
    download_url="https://github.com/ssrl-px/iota/releases",
    license="BSD",
    install_requires=["matplotlib", "numpy", "six", "wxpython"],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    entry_points={
        "console_scripts": [
            "iota = iota.command_line.iota_gui_launch:entry_point",
            "iota.filter_pickles = iota.command_line.iota_filter_pickles:entry_point",
            "iota.run = iota.command_line.iota_run:entry_point",
            "iota.single_image = iota.command_line.iota_single_image:entry_point",
            "iota.track_images = iota.command_line.iota_track_images:entry_point",
        ],
        "libtbx.dispatcher.script": [
            "iota = iota",
            "iota.filter_pickles = iota.filter_pickles",
            "iota.run = iota.run",
            "iota.single_image = iota.single_image",
            "iota.track_images = iota.track_images",
        ],
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
