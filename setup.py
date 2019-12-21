from __future__ import absolute_import, division, print_function

import setuptools

command_line_scripts = [
"iota_filter_pickles",
"iota_gui_launch",
"iota_run",
"iota_single_image",
"iota_track_images",
]

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
        "console_scripts": [
#            "iota.double_gauss_visualizer,  is prime
        "iota.single_image = iota.command_line.iota_single_image",
#"iota.filter_pickles"          , "iota.run" , "iota.track_images",
],
        "libtbx.dispatcher.script": ["iota.single_image = iota.single_image"],
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
