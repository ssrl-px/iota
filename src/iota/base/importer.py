from __future__ import absolute_import, division, print_function

import iota.utils.input_finder

"""
Author      : Lyubimov, A.Y.
Created     : 02/08/2021
Last Changed: 02/08/2021
Description : IOTA Importer base class
"""

import os

from dxtbx.model.experiment_list import ExperimentListFactory as ExLF

from iota.utils import utils


class SingleImageBase(object):
    """Base class for containing all info (including data) for single image;
    can also save to file if necessary or perform conversion if necessary."""

    def __init__(self, imgpath, idx=None, img_idx=0, is_multi_image=False):
        self.img_path = imgpath
        self.obj_path = None
        self.obj_file = None
        self.int_path = None
        self.int_file = None
        self.viz_path = None
        self.int_log = None
        self.experiments = None
        self.is_multi_image = is_multi_image

        # Image index (img_index) refers to the ranking index of the image
        # within a multi-image file (e.g. HDF5); input index (input_index) refers
        # to the ranking index of the image within the full input list.
        self.img_index = img_idx
        if idx:
            self.input_index = idx
        else:
            self.input_index = 0

        # Processing properties
        self.status = None
        self.fail = None
        self.log_info = []
        self.errors = []

        # Final statistics (may add stuff to dictionary later, as needed)
        self.final = {
            "img": None,  # Image filepath
            "img_idx": 0,  # Image index
            "final": None,  # Integrated pickle
            "observations": None,  # Miller array
            "info": "",  # Information (general)
            "a": 0,
            "b": 0,
            "c": 0,  # Cell edges
            "alpha": 0,
            "beta": 0,
            "gamma": 0,  # Cell angles
            "sg": None,  # Space group
            "spots": 0,  # Found spots
            "indexed": 0,  # Indexed reflections
            "integrated": 0,  # Integrated reflections
            "strong": 0,  # Strong int. reflections
            "res": 0,
            "lres": 0,  # Resolution, low res.
            "mos": 0,
            "epv": 0,  # Mosaicity, EPV
            "wavelength": 0,  # Wavelength
            "distance": 0,  # Distance
            "beamX": 0,
            "beamY": 0,  # Beam XY (mm)
            "img_size": (0, 0),  # Image size (pixels)
            "pixel_size": 0,  # Pixel Size
            "gain": 0,  # Detector gain
        }


class ImageImporterBase:
    """Base class for image importer, which will:

    1. read an image file and extract data and info
    2. apply any modifications if requested / necessary
    3. output an experiment list or file for processing
    """

    def __init__(self, init=None, info=None, write_output=True):

        self.init = init if init else None
        self.info = info if info else None
        assert init or info

        self.img_type = None
        self.filepath = None
        self.modify = False
        self.write_output = write_output

    def instantiate_image_object(
        self, filepath, idx=None, img_idx=0, is_multi_image=False
    ):
        """Override to instantiate a SingleImage object for current backend.

        :param filepath: path to image file
        :param idx: input index (ranking index for image in full input list)
        :param img_idx: image index (index of image in multi-image file)
        :return: an image object
        """
        self.img_object = SingleImageBase(
            imgpath=filepath, idx=idx, img_idx=img_idx, is_multi_image=is_multi_image
        )

    def load_image_file(self, filepath, experiments=None):
        """Loads experiment list and populates image information dictionary
        (can override to load images for old-timey HA14 processing)

        :param filepath: path to raw diffraction image (or pickle!)
        :param experiments: an ExperimentList object can be passed to this function
        :return: experiment list, error message (if any)
        """
        if not experiments:
            try:
                experiments = ExLF.from_filenames(filenames=[filepath])
            except Exception as e:
                error = "IOTA IMPORTER ERROR: Import failed! {}".format(e)
                print(error)
                return None, error

        # Load image information from experiment object
        try:
            imgset = experiments.imagesets()[0]
            beam = imgset.get_beam()
            s0 = beam.get_s0()
            detector = imgset.get_detector()[0]

            self.img_object.final["pixel_size"] = detector.get_pixel_size()[0]
            self.img_object.final["img_size"] = detector.get_image_size()
            self.img_object.final["beamX"] = detector.get_beam_centre(s0)[0]
            self.img_object.final["beamY"] = detector.get_beam_centre(s0)[1]
            self.img_object.final["gain"] = detector.get_gain()
            self.img_object.final["distance"] = detector.get_distance()
            self.img_object.final["wavelength"] = beam.get_wavelength()

        except Exception as e:
            error = "IOTA IMPORTER ERROR: Information extraction failed! {}".format(e)
            print(error)
            return experiments, error

        return experiments, None

    def modify_image(self, data=None):
        """Override for specific backend needs (i.e. convert, mask, and/or
        square images and output pickles for HA14."""
        return data, None

    def calculate_parameters(self, experiments=None):
        """Override to perform calculations of image-specific parameters.

        :param experiments: Experiment list with image info
        :return: experiment list, error message
        """
        return experiments, None

    def update_log(self, data=None, status="imported", msg=None):
        if not data:
            self.img_object.log_info.append("\n{}".format(msg))
            self.img_object.status = "failed import"
            self.img_object.fail = "failed import"
        else:
            beamx = "BEAM_X = {:<4.2f}, ".format(self.img_object.final["beamX"])
            beamy = "BEAM_Y = {:<4.2f}, ".format(self.img_object.final["beamY"])
            pixel = "PIXEL_SIZE = {:<8.6f}, " "".format(
                self.img_object.final["pixel_size"]
            )
            size = "IMG_SIZE = {:<4} X {:<4}, " "".format(
                self.img_object.final["img_size"][0],
                self.img_object.final["img_size"][1],
            )
            dist = "DIST = {}".format(self.img_object.final["distance"])
            info = ["Parameters      :", beamx, beamy, pixel, size, dist]
            self.img_object.log_info.append("".join(info))
            self.img_object.status = status
            self.img_object.fail = None

    def prep_output(self):
        """Assign output paths to image object (will be used by various modules
        to write out files in appropriate locations)"""

        # Get bases
        input_base = self.info.input_base if self.info else self.init.input_base
        obj_base = self.info.obj_base if self.info else self.init.obj_base
        fin_base = self.info.fin_base if self.info else self.init.fin_base
        log_base = self.info.log_base if self.info else self.init.log_base
        dials_log_base = self.info.dials_log_base if self.info else None
        viz_base = self.info.viz_base if self.info else self.init.viz_base

        # Set prefix (image index in multi-image files, or None otherwise)
        if self.img_object.is_multi_image:
            image_index = self.img_object.img_index
        else:
            image_index = None

        # Object path (may not need)
        self.img_object.obj_path = utils.make_image_path(
            self.img_object.img_path, input_base, obj_base
        )
        fname = utils.make_filename(
            prefix=image_index, path=self.img_object.img_path, new_ext="int"
        )
        self.img_object.obj_file = os.path.join(self.img_object.obj_path, fname)

        # DIALS process filepaths
        # Indexed reflections
        ridx_path = utils.make_filename(
            prefix=image_index,
            path=self.img_object.img_path,
            suffix="indexed",
            new_ext="refl",
        )
        self.img_object.ridx_path = os.path.join(self.img_object.obj_path, ridx_path)

        # Spotfinding (strong) reflections
        rspf_path = utils.make_filename(
            prefix=image_index,
            path=self.img_object.img_path,
            suffix="strong",
            new_ext="refl",
        )
        self.img_object.rspf_path = os.path.join(self.img_object.obj_path, rspf_path)

        # Refined experiments
        eref_path = utils.make_filename(
            prefix=image_index,
            path=self.img_object.img_path,
            suffix="refined",
            new_ext="expt",
        )
        self.img_object.eref_path = os.path.join(self.img_object.obj_path, eref_path)

        # Integrated experiments
        eint_path = utils.make_filename(
            prefix=image_index,
            path=self.img_object.img_path,
            suffix="integrated",
            new_ext="expt",
        )
        self.img_object.eint_path = os.path.join(self.img_object.obj_path, eint_path)

        # Integrated reflections
        rint_path = utils.make_filename(
            prefix=image_index,
            path=self.img_object.img_path,
            suffix="integrated",
            new_ext="refl",
        )
        self.img_object.rint_path = os.path.join(self.img_object.obj_path, rint_path)

        # Final integration pickle path
        self.img_object.int_path = utils.make_image_path(
            self.img_object.img_path, input_base, fin_base
        )
        fname = utils.make_filename(
            path=self.img_object.img_path,
            prefix="int",
            suffix=image_index,
            new_ext="pickle",
        )
        self.img_object.int_file = os.path.join(self.img_object.int_path, fname)

        # Processing log path for image
        self.img_object.log_path = utils.make_image_path(
            self.img_object.img_path, input_base, log_base
        )
        fname = utils.make_filename(
            prefix=image_index, path=self.img_object.img_path, new_ext="log"
        )
        self.img_object.int_log = os.path.join(self.img_object.log_path, fname)

        # DIALS log path for image
        if dials_log_base:
            self.img_object.dials_log_path = utils.make_image_path(
                self.img_object.img_path, input_base, dials_log_base
            )
            fname = utils.make_filename(
                prefix=image_index, path=self.img_object.img_path, new_ext="log"
            )
            self.img_object.dials_log = os.path.join(
                self.img_object.dials_log_path, fname
            )

        # Visualization path (may need to deprecate)
        self.img_object.viz_path = utils.make_image_path(
            self.img_object.img_path, input_base, viz_base
        )
        fname = utils.make_filename(
            prefix="int",
            suffix=image_index,
            path=self.img_object.img_path,
            new_ext="png",
        )
        self.viz_file = os.path.join(self.img_object.viz_path, fname)

        # Make paths if they don't exist already
        for path in [
            self.img_object.obj_path,
            self.img_object.int_path,
            self.img_object.log_path,
            self.img_object.viz_path,
            self.img_object.dials_log_path,
        ]:
            try:
                os.makedirs(path)
            except OSError:
                pass

        # Populate the 'final' dictionary
        self.img_object.final["final"] = self.img_object.int_file
        self.img_object.final["img"] = self.img_object.img_path

    def import_image(self, input_entry):
        """Image importing: read file, modify if requested, make experiment list
        :return: An image object with info and experiment list
        """

        # Interpret input
        expr = None
        is_multi_image = False
        if type(input_entry) in (list, tuple):
            idx = int(input_entry[0])
            filepath = str(input_entry[1])
            img_idx = int(input_entry[2])
            if len(input_entry) == 4:
                expr = input_entry[3]
                is_multi_image = True
        elif type(input_entry) == str:
            idx = None
            filepath = input_entry
            img_idx = 0
        else:
            raise iota.utils.input_finder.InputError(
                "IOTA IMPORT ERROR: Unrecognized input -- {}" "".format(input_entry)
            )

        # Instantiate image object
        self.instantiate_image_object(
            filepath, idx, img_idx, is_multi_image=is_multi_image
        )

        # Generate output paths
        if self.write_output:
            self.prep_output()

        # Load image (default is experiment list, override for HA14-style pickling)
        self.experiments, error = self.load_image_file(
            filepath=filepath, experiments=expr
        )

        # Log initial image information
        self.img_object.log_info.append("\n{:-^100}\n".format(filepath))
        self.update_log(data=self.experiments, msg=error)

        # Stop there if data did not load
        if not self.experiments:
            return self.img_object, error

        # Modify image as per backend (or not)
        if self.modify:
            self.experiments, error = self.modify_image(data=self.experiments)
            if not self.experiments:
                self.update_log(data=None, msg=error)
                return self.img_object, error
            else:
                self.update_log(data=self.experiments, msg=error, status="converted")

        # Calculate image-specific parameters (work in progress)
        self.experiments, error = self.calculate_parameters(
            experiments=self.experiments
        )
        if error:
            self.update_log(data=self.experiments, msg=error)

        # Finalize and output
        self.img_object.experiments = self.experiments
        self.img_object.status = "imported"

        return self.img_object, None

    def make_image_object(self, input_entry):
        """Run image importer (override as needed)"""
        img_object, error = self.import_image(input_entry=input_entry)
        if error:
            print(error)
        return img_object

    def run(self, input_entry):
        return self.make_image_object(input_entry)

# -- end
