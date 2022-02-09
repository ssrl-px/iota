from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 02/08/2021
Last Changed: 02/08/2021
Description : IOTA Processor base class
"""

import time
from threading import Thread

from dxtbx.model.experiment_list import ExperimentListFactory as ExLF
from libtbx.easy_mp import parallel_map


class ProcessingBase(Thread):
    """Base class for submitting processing jobs to multiple cores."""

    def __init__(self, *args, **kwargs):
        Thread.__init__(self, *args, **kwargs)

        # Set attributes from remaining kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def create_iterable(self, input_list):
        return [[i, len(input_list) + 1, str(j)] for i, j in enumerate(input_list, 1)]

    def import_and_process(self, input_entry):
        img_object = self.importer.run(input_entry)
        if img_object.status == "imported":
            with util.Capturing() as junk_output:
                img_object = self.integrator.run(img_object)

        # Update main log
        if hasattr(self.info, "logfile"):
            main_log_entry = "\n".join(img_object.log_info)
            util.main_log(self.info.logfile, main_log_entry)
            util.main_log(self.info.logfile, "\n{:-^100}\n".format(""))

        # Set status to final
        img_object.status = "final"
        return img_object

    def callback(self, result):
        """Override for custom callback behavior (or not)"""
        return result

    def run_process(self, iterable):
        # Create ExperimentList objects from image paths (doing it in the processor
        # because I can't have Python objects in the INFO JSON file)
        adj_iterable = []
        crystal = None
        for entry in iterable:
            path = str(entry[1])
            if path.endswith(".h5"):
                exp_idx = entry[0]
                img_idx = entry[2]

                # Generate a list of single images if at img_idx = 0
                if img_idx == 0:
                    imageseqs = []
                    exps = ExLF.from_filenames(filenames=[path])
                    crystal = exps[0].crystal

                    # flatten all imagesets into a single imagesequence
                    for iset in exps.imagesets():
                        if iset.size() == 1:
                            imageseqs.append(iset)
                        else:
                            for i in range(iset.size()):
                                one_image = imageseq.partial_set(i, i + 1)
                                imageseqs.append(i)

                # Create ExperimentList object from extracted imageset
                current_image = imageseqs[img_idx]
                one_exp = ExLF.from_imageset_and_crystal(
                    imageset=current_image, crystal=crystal
                )
                adj_iterable.append([exp_idx, path, img_idx, one_exp])
            else:
                # Create ExperimentList object from CBF
                expr = ExLF.from_filenames(filenames=[path])
                exp_entry = [entry[0], entry[1], 0, expr]
                adj_iterable.append(exp_entry)

        # Run a multiprocessing job
        img_objects = parallel_map(
            iterable=adj_iterable,
            func=self.import_and_process,
            callback=self.callback,
            processes=self.params.mp.n_processors,
        )
        return img_objects

    def run_analysis(self):
        """Run analysis of integrated images."""
        from iota.components.iota_analysis import Analyzer

        analysis = Analyzer(info=self.info, params=self.params)
        self.info = analysis.run_all()
        self.info.export_json()

    def process(self):
        """Run importer and/or processor."""
        img_objects = self.run_process(iterable=self.info.unprocessed)
        self.info.finished_objects = img_objects
        self.run_analysis()

    def run(self):
        self.process()

    @classmethod
    def for_new_run(cls, paramfile, run_no, *args, **kwargs):
        from iota.components.iota_image import ImageImporter as Importer
        from iota.components.iota_processing import IOTAImageProcessor as Integrator

        # Initialize processing parameters
        from iota.components.iota_init import initialize_processing

        cls.info, cls.params = initialize_processing(paramfile, run_no)

        cls.importer = Importer(info=cls.info)
        cls.integrator = Integrator(iparams=cls.params)

        return cls(*args, **kwargs)

    @classmethod
    def for_existing_run(cls, info, *args, **kwargs):

        from iota.components.iota_image import ImageImporter as Importer
        from iota.components.iota_processing import IOTAImageProcessor as Integrator

        # Initialize processing parameters
        from iota.components.iota_init import resume_processing

        cls.info, cls.params = resume_processing(info)

        cls.importer = Importer(info=cls.info)
        cls.integrator = Integrator(iparams=cls.params)

        return cls(*args, **kwargs)

    @classmethod
    def for_single_image(
        cls, info, params, action_code="spotfinding", verbose=False, *args, **kwargs
    ):

        from iota.components.iota_image import ImageImporter
        from iota.components.iota_processing import IOTAImageProcessor

        # Initialize processing parameters
        cls.info = info
        cls.params = params
        cls.action_code = action_code
        cls.verbose = verbose

        cls.importer = ImageImporter(info=cls.info, write_output=False)
        cls.integrator = IOTAImageProcessor(
            iparams=cls.params,
            write_pickle=False,
            write_logs=False,
            last_stage=action_code,
        )

        return cls(*args, **kwargs)
