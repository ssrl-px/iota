from __future__ import absolute_import, division, print_function

# First need to find the location of setup.py.
# We can't use '__file__' because of libtbx,
# and 'import iota' doesn't exist yet.
import os
import libtbx.load_env

iota_dir = libtbx.env.find_in_repositories(relative_path="iota", test=os.path.exists)

# Run setup.py
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "setup.py", "develop"], cwd=iota_dir
)
if result.returncode:
    exit("Error during IOTA configuration")

# import dials.precommitbx.nagger
# dials.precommitbx.nagger.nag()
