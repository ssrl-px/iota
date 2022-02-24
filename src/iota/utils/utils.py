from __future__ import absolute_import, division, print_function

from iota.utils.input_finder import InputFinder
from six.moves import range, zip

"""
Author      : Lyubimov, A.Y.
Created     : 12/19/2016
Last Changed: 11/21/2019
Description : Module with basic utilities of broad applications in IOTA
"""

import os
import sys
import wx

from cctbx import miller

assert miller
from libtbx import easy_pickle as ep

# for Py2/3 compatibility
if sys.version_info[0] == 3:
    from io import StringIO
else:
    from six.moves import StringIO


# For testing
import time

assert time

# Platform-specific stuff
# TODO: Will need to test this on Windows at some point
if wx.Platform == "__WXGTK__":
    plot_font_size = 9
    norm_font_size = 9
    button_font_size = 11
    LABEL_SIZE = 11
    CAPTION_SIZE = 9
    python = "python"
elif wx.Platform == "__WXMAC__":
    plot_font_size = 9
    norm_font_size = 12
    button_font_size = 14
    LABEL_SIZE = 14
    CAPTION_SIZE = 12
    python = "Python"
elif wx.Platform == "__WXMSW__":
    plot_font_size = 9
    norm_font_size = 9
    button_font_size = 11
    LABEL_SIZE = 11
    CAPTION_SIZE = 9
    python = "Python"  # TODO: make sure it's right!

# --------------------------- Miscellaneous Utils ---------------------------- #


def noneset(value):
    if value == "":
        return "None"
    elif "none" in str(value).lower():
        return "None"
    elif value is None:
        return "None"
    else:
        return value


def makenone(value):
    if str(value).lower() in ("none", ""):
        return None
    else:
        return str(value)


class UnicodeCharacters:
    def __init__(self):
        self.alpha = u"\N{GREEK SMALL LETTER ALPHA}".encode("utf-8")
        self.beta = u"\N{GREEK SMALL LETTER BETA}".encode("utf-8")
        self.gamma = u"\N{GREEK SMALL LETTER GAMMA}".encode("utf-8")
        self.sigma = u"\N{GREEK SMALL LETTER SIGMA}".encode("utf-8")


class WxFlags:
    def __init__(self):
        self.stack = wx.TOP | wx.RIGHT | wx.LEFT
        self.expand = wx.TOP | wx.RIGHT | wx.LEFT | wx.EXPAND


class Capturing(list):
    """Class used to capture stdout from cctbx.xfel objects.

    Saves output in appendable list for potential logging.
    """

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self._io_stdout = StringIO()
        sys.stderr = self._io_stderr = StringIO()
        return self

    def __exit__(self, *args):
        bytes_str = self._io_stdout.getvalue()
        stdout_lines = bytes_str.splitlines()  # .decode('UTF-8').splitlines()
        self.extend(stdout_lines)
        sys.stdout = self._stdout

        bytes_str = self._io_stderr.getvalue()
        stderr_lines = bytes_str.splitlines()  # .decode('UTF-8').splitlines()
        self.extend(stderr_lines)
        sys.stderr = self._stderr


def convert_phil_to_text(phil, phil_file=None, att_level=0):
    """Reads in a PHIL object and converts it to plain text; optionally writes
    out to text file if filepath is provided.

    :param phil: PHIL object
    :param phil_file: absolute filepath for text file with parameters
    :return: PHIL text string
    """

    txt_out = phil.as_str(attributes_level=att_level)

    if phil_file:
        with open(phil_file, "w") as pf:
            pf.write(txt_out)

    return txt_out


def get_mpi_rank_and_size():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # each process in MPI has a unique id, 0-indexed
    size = comm.Get_size()  # size: number of processes running in this job
    return rank, size


def main_log(logfile, entry, print_tag=False):
    """Write main log (so that I don't have to repeat this every time).

    All this is necessary so that I don't have to use the Python logger
    module, which creates a lot of annoying crosstalk with other
    cctbx.xfel modules.
    """
    if logfile is not None:
        with open(logfile, "a") as lf:
            lf.write("{}\n".format(entry))

    if print_tag:
        print(entry)


def set_base_dir(dirname=None, sel_flag=False, out_dir=None, get_run_no=False):
    """Generates a base folder for converted pickles and/or integration
    results; creates subfolder numbered one more than existing."""
    if out_dir is None and dirname is not None:
        path = os.path.abspath(os.path.join(os.curdir, dirname))
    elif out_dir is not None and dirname is None:
        path = os.path.abspath(out_dir)
    elif out_dir is None and dirname is None:
        path = os.path.abspath(os.curdir)
    else:
        path = os.path.join(os.path.abspath(out_dir), dirname)

    run_no = 1
    if os.path.isdir(path):
        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        dirnums = [int(d) for d in dirs if d.isdigit()]
        if len(dirnums) > 0:
            n_top = max(dirnums)
            if sel_flag:
                run_no = n_top
            else:
                run_no = n_top + 1

    new_path = "{}/{:03d}".format(path, run_no)

    if get_run_no:
        return new_path, run_no
    else:
        return new_path


def find_base_dir(dirname):
    """Function to determine the current folder name."""

    def check_dirname(path, subdirname):
        if os.path.isdir(os.path.join(path, subdirname)):
            try:
                int(subdirname)
                return True
            except ValueError:
                return False
        else:
            return False

    path = os.path.abspath(os.path.join(os.curdir, dirname))
    if os.path.isdir(path):
        if len(os.listdir(path)) > 0:
            dirs = [int(i) for i in os.listdir(path) if check_dirname(path, i)]
            found_path = "{}/{:03d}".format(path, max(dirs))
        else:
            found_path = path
    else:
        found_path = os.curdir
    return found_path


def make_image_path(raw_img, input_base, base_path):
    """Makes path for output images."""
    path = os.path.dirname(raw_img)
    relpath = os.path.relpath(path, input_base)
    if relpath == ".":
        dest_folder = base_path
    else:
        dest_folder = os.path.join(base_path, relpath)
    return os.path.normpath(dest_folder)
    # return dest_folder


def make_filename(path, prefix=None, suffix=None, new_ext=None):
    bname = os.path.basename(path)
    ext = bname.split(os.extsep)[-1]
    if ext.isdigit():
        filename = bname
    else:
        fn_list = bname.split(".")
        filename = ".".join(fn_list[0:-1])
    if prefix is not None:
        filename = "{}_{}".format(prefix, filename)
    if suffix is not None:
        filename = "{}_{}".format(filename, suffix)
    if new_ext is not None:
        filename = "{}.{}".format(filename, new_ext)
    return filename


def iota_exit(silent=False, msg=None):
    if not silent:
        from iota import iota_version, now

        if msg:
            print(msg)
        print("\n\nIOTA version {0}".format(iota_version))
        print("{}\n".format(now))
    sys.exit()


def get_size(obj, seen=None):
    """Recursively finds size of objects (by Wissam Jarjoui at SHippo,
    https://goshippo.com/blog/measure-real-size-any-python-object/)"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size


# ------------------------------ Input Finder -------------------------------- #


ginp = InputFinder()


class ObjectFinder(object):
    """A class for finding pickled IOTA image objects and reading in their
    contents; outputs a list of Python objects containing information about
    individual images, including a list of integrated intensities."""

    def __init__(self):
        """Constructor."""
        pass

    def find_objects(
        self, obj_folder, read_object_files=None, find_old=False, finished_only=False
    ):
        """Seek and import IOTA image objects.

        :param finished_only:
        :param obj_folder: path to objects (which can be in subfolders)
        :param read_object_files: list of already-read-in objects
        :param find_old: find all objects in folder, regardless of other settings
        :return: list of image objects
        """
        if find_old:
            min_back = None
        else:
            min_back = -1

        # Find objects and filter out already-read objects if any
        object_files = ginp.get_input_from_folder(
            obj_folder, ext_only="int", min_back=min_back
        )

        if read_object_files is not None:
            new_object_files = list(set(object_files) - set(read_object_files))
        else:
            new_object_files = object_files

        # For backwards compatibility, read and append observations to objects
        new_objects = [self.read_object_file(i) for i in new_object_files]
        new_finished_objects = [
            i for i in new_objects if i is not None and i.status == "final"
        ]

        if finished_only:
            return new_finished_objects
        else:
            return new_objects

    def read_object_file(self, filepath):
        """Load pickled image object; if necessary, extract observations from
        the image pickle associated with object, and append to object.

        :param filepath: path to image object file
        :return: read-in (and modified) image object
        """
        try:
            object = ep.load(filepath)
            if object.final["final"] is not None:
                pickle_path = object.final["final"]
                if os.path.isfile(pickle_path):
                    pickle = ep.load(pickle_path)
                    object.final["observations"] = pickle["observations"][0]
            return object
        except Exception as e:
            print("OBJECT_IMPORT_ERROR for {}: {}".format(filepath, e))
            return None


gobj = ObjectFinder()

# ---------------------------------- Other ----------------------------------- #


class RadAverageCalculator(object):
    def __init__(self, image=None, experiments=None):
        if image is None and experiments is None:
            print("ERROR: Need image or experiments for Radial Average Calculator")
            return
        if experiments is None:
            from dxtbx.model.experiment_list import ExperimentListFactory

            self.experiments = ExperimentListFactory.from_filenames([image])[0]
        else:
            self.experiments = experiments

    def make_radial_average(self, num_bins=None, hires=None, lowres=None):
        from dials.algorithms.background import RadialAverage

        imageset = self.experiments.extract_imagesets()[0]
        beam = imageset.get_beam()
        detector = imageset.get_detector()
        scan_range = (0, len(imageset))

        summed_data = None
        summed_mask = None
        for i in range(*scan_range):
            data = imageset.get_raw_data(i)
            mask = imageset.get_mask(i)
            assert isinstance(data, tuple)
            assert isinstance(mask, tuple)
            if summed_data is None:
                summed_mask = mask
                summed_data = data
            else:
                summed_data = [sd + d for sd, d in list(zip(summed_data, data))]
                summed_mask = [sm & m for sm, m in list(zip(summed_mask, mask))]

        if num_bins is None:
            num_bins = int(sum(sum(p.get_image_size()) for p in detector) / 50)
        if lowres is not None:
            vmin = (1 / lowres) ** 2
        else:
            vmin = 0
        if hires is not None:
            vmax = (1 / hires) ** 2
        else:
            vmax = (1 / detector.get_max_resolution(beam.get_s0())) ** 2

        # Compute the radial average
        radial_average = RadialAverage(beam, detector, vmin, vmax, num_bins)
        for d, m in list(zip(summed_data, summed_mask)):
            radial_average.add(d.as_double() / (scan_range[1] - scan_range[0]), m)
        mean = radial_average.mean()
        reso = radial_average.inv_d2()
        return mean, reso


class IOTATermination(Exception):
    def __init__(self, termination):
        Exception.__init__(self, termination)


