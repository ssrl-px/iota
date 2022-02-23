from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 02/08/2021
Last Changed: 02/08/2021
Description : IOTA Processor base class
"""

import time
import copy
from threading import Thread

from dxtbx.model.experiment_list import ExperimentList, ExperimentListFactory as ExLF
from libtbx.easy_mp import parallel_map
from libtbx.phil import parse
from libtbx.utils import Abort, Sorry

from dials.array_family import flex

from iota.utils import utils

control_phil_str = """
  input {
    file_list = None
      .type = path
      .help = Path to a text file with a list of images
    glob = None
      .type = str
      .help = For large, multi-file datasets, specify the paths using wildcards (e.g. *.cbf)
      .multiple = True
    image_tag = None
      .type = str
      .multiple = True
      .help = Only process images with these tag(s). For single-image files (like CBFs or SMVs), the image \
              tag for each file is the file name. For multi-image files like HDF5, the image tag is        \
              filename_imagenumber (including leading zeros). Use show_image_tags=True to see the list     \
              of image tags that will be used for a dataset.
    show_image_tags = False
      .type = bool
      .help = Show the set of image tags that would be used during processing. To process subsets of image \
              files, use these tags with the image_tag parameter.
    max_images = None
      .type = int
      .help = Limit total number of processed images to max_images
    ignore_gain_mismatch = False
      .type = bool
      .expert_level = 3
      .help = Detector gain should be set on the detector models loaded from the images or in the \
              processing parameters, not both. Override the check that this is true with this flag. \
  }

  dispatch {
    pre_import = False
      .type = bool
      .expert_level = 2
      .help = If True, before processing import all the data. Needed only if processing \
              multiple multi-image files at once (not a recommended use case)
    process_percent = None
      .type = int(value_min=1, value_max=100)
      .help = Percent of events to process
    refine = False
      .expert_level = 2
      .type = bool
      .help = If True, after indexing, refine the experimental models
    squash_errors = True
      .expert_level = 2
      .type = bool
      .help = If True, if an image fails to process, continue to the next image. \
              otherwise, halt processing and show the error.
    find_spots = True
      .expert_level = 2
      .type = bool
      .help = Whether to do spotfinding. Needed for indexing/integration
    index = True
      .expert_level = 2
      .type = bool
      .help = Attempt to index images. find_spots also needs to be True for this to work
    integrate = True
      .expert_level = 2
      .type = bool
      .help = Integrate indexed images. Ignored if index=False or find_spots=False
    coset = False
      .expert_level = 2
      .type = bool
      .help = Within the integrate dispatcher, integrate a sublattice coset intended to represent \
              negative control spots with no Bragg diffraction.
    hit_finder{
      enable = True
        .type = bool
        .help = Whether to do hitfinding. hit_finder=False: process all images
      minimum_number_of_reflections = 16
        .type = int
        .help = If the number of strong reflections on an image is less than this, and \
                 the hitfinder is enabled, discard this image.
      maximum_number_of_reflections = None
       .type = int
       .help = If specified, ignores images with more than this many number of reflections
    }
  }

  output {
    output_dir = .
      .type = str
      .help = Directory output files will be placed
    composite_output = True
      .type = bool
      .help = If True, save one set of experiment/reflection files per process, where each is a \
              concatenated list of all the successful events examined by that process. \
              If False, output a separate experiment/reflection file per image (generates a \
              lot of files).
    logging_dir = None
      .type = str
      .help = Directory output log files will be placed
    experiments_filename = None
      .type = str
      .help = The filename for output experiments. For example, %s_imported.expt
    strong_filename = None
      .type = str
      .help = The filename for strong reflections from spot finder output. For example: \
              %s_strong.refl
    indexed_filename = %s_indexed.refl
      .type = str
      .help = The filename for indexed reflections.
    refined_experiments_filename = %s_refined.expt
      .type = str
      .help = The filename for saving refined experimental models
    integrated_filename = %s_integrated.refl
      .type = str
      .help = The filename for final integrated reflections.
    integrated_experiments_filename = %s_integrated.expt
      .type = str
      .help = The filename for saving final experimental models.
    coset_filename = %s_coset%d.refl
      .type = str
      .help = The filename for final coset reflections.
    coset_experiments_filename = %s_coset%d.expt
      .type = str
      .help = The filename for saving final coset experimental models.
    profile_filename = None
      .type = str
      .help = The filename for output reflection profile parameters
    integration_pickle = int-%d-%s.pickle
      .type = str
      .help = Filename for cctbx.xfel-style integration pickle files
  }

  mp {
    method = *multiprocessing sge lsf pbs mpi
      .type = choice
      .help = "The multiprocessing method to use"
    nproc = 1
      .type = int(value_min=1)
      .help = "The number of processes to use."
    composite_stride = None
      .type = int
      .help = For MPI, if using composite mode, specify how many ranks to    \
              aggregate data from.  For example, if you have 100 processes,  \
              composite mode will output N*100 files, where N is the number  \
              of file types (expt, refl, etc). If you specify stride = 25, \
              then each group of 25 process will send their results to 4     \
              processes and only N*4 files will be created. Ideally, match   \
              stride to the number of processors per node.
    debug
      .expert_level = 2
    {
      cProfile = False
        .type = bool
        .help = Enable code profiling. Profiling file will be available in  \
                the debug folder. Use (for example) runsnake to visualize   \
                processing performance
      output_debug_logs = True
        .type = bool
        .help = Whether to write debugging information for every image      \
                processed
    }
  }
"""

dials_phil_str = """
  input {
    reference_geometry = None
      .type = str
      .help = Provide an models.expt file with exactly one detector model. Data processing will use \
              that geometry instead of the geometry found in the image headers.
    sync_reference_geom = True
      .type = bool
      .help = ensures the reference hierarchy agrees with the image format
  }

  output {
    shoeboxes = True
      .type = bool
      .help = Save the raw pixel values inside the reflection shoeboxes during spotfinding.
  }

  include scope dials.util.options.geometry_phil_scope
  include scope dials.algorithms.spot_finding.factory.phil_scope
  include scope dials.algorithms.indexing.indexer.phil_scope
  indexing {
      include scope dials.algorithms.indexing.lattice_search.basis_vector_search_phil_scope
  }
  include scope dials.algorithms.refinement.refiner.phil_scope
  include scope dials.algorithms.integration.integrator.phil_scope
  include scope dials.algorithms.profile_model.factory.phil_scope
  include scope dials.algorithms.spot_prediction.reflection_predictor.phil_scope
  include scope dials.algorithms.integration.stills_significance_filter.phil_scope

  indexing {
    stills {
      method_list = None
        .type = strings
        .help = List of indexing methods. If indexing fails with first method, indexing will be \
                attempted with the next, and so forth
    }
  }

  integration {
    include scope dials.algorithms.integration.kapton_correction.absorption_phil_scope
    coset {
      transformation = 6
        .type = int(value_min=0, value_max=6)
        .multiple = False
        .help = The index number(s) of the modulus=2 sublattice transformation(s) used to produce distince coset results. \
                0=Double a, 1=Double b, 2=Double c, 3=C-face centering, 4=B-face centering, 5=A-face centering, 6=Body centering \
                See Sauter and Zwart, Acta D (2009) 65:553
    }

    integration_only_overrides {
      trusted_range = None
        .type = floats(size=2)
        .help = "Override the panel trusted range (underload and saturation) during integration."
        .short_caption = "Panel trusted range"
    }
  }

  profile {
    gaussian_rs {
      parameters {
        sigma_b_cutoff = 0.1
          .type = float
          .help = Maximum sigma_b before the image is rejected
      }
    }
  }
"""

program_defaults_phil_str = """
indexing {
  method = fft1d
}
refinement {
  parameterisation {
    auto_reduction {
      min_nref_per_parameter = 1
      action = fix
    }
    beam.fix = all
    detector.fix = all
  }
  reflections {
    weighting_strategy.override = stills
    outlier.algorithm = null
  }
}
integration {
  integrator = stills
  profile.fitting = False
  background {
    algorithm = simple
    simple {
      outlier.algorithm = plane
      model.algorithm = linear2d
    }
  }
}
profile.gaussian_rs.min_spots.overall = 0
"""

phil_scope = parse(control_phil_str + dials_phil_str, process_includes=True).fetch(
    parse(program_defaults_phil_str)
)


class Processor(object):
    """ Base processor class derived from the DIALS stills processor """

    # TODO: Break up into one-action functions
    # TODO: Figure out logging that can be turned on and off
    # TODO: Implement hopper functionality
    # TODO: Bring closer to an API than a subclassable class

    def __init__(self, params, verbose=False):
        self.params = params
        self.known_crystal_models = None
        self.verbose = verbose

    def find_spots(self, experiments, reset_z=False):
        """Find strong spots
        @param experiments - Experiment list object
        """
        st = time.time()
        if self.verbose:
            print("*" * 80)
            print("Finding Strong Spots")
            print("*" * 80)

        # Find spots
        observed = flex.reflection_table.from_observations(
            experiments, self.params, is_stills=True
        )
        if self.verbose:
            print("\n{} strong spots found".format(observed.size()))

        # Reset z-coordiantes for dials.image_viewer
        if reset_z:
            self.reset_z_coordinates(reflections=observed)
            print("Z-coordinates of reflections reset for dials.image_viewer")

        # Save strong spots to file if there's a filename
        print("\n" + "-" * 80)
        print("Strong spots saved to {}".format(self.params.output.strong_filename))
        if self.params.output.strong_filename:
            self.save_reflections(observed, self.params.output.strong_filename)

        if self.verbose:
            print("\nSpotfinding time = {.2f} seconds".format(time.time() - st))
        return observed

    @staticmethod
    def reset_z_coordinates(reflections):
        """Reset z coordinates for dials.image_viewer; see Issues #226 for details
        @param reflections - A reflection table
        """
        xyzobs = reflections["xyzobs.px.value"]
        for i in range(len(xyzobs)):
            xyzobs[i] = (xyzobs[i][0], xyzobs[i][1], 0)
        bbox = reflections["bbox"]
        for i in range(len(bbox)):
            bbox[i] = (bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3], 0, 1)
        return reflections

    def index(self, experiments, reflections):
        from dials.algorithms.indexing.indexer import Indexer

        # deepcopy params because they will be modified here
        params = copy.deepcopy(self.params)

        # don't do scan-varying refinement during indexing
        params.refinement.parameterisation.scan_varying = False

        if params.indexing.stills.method_list is None:
            idxr = Indexer.from_parameters(
                reflections,
                experiments,
                known_crystal_models=self.known_crystal_models,
                params=params,
            )
            idxr.index()
        else:
            indexing_error = None
            for method in params.indexing.stills.method_list:
                params.indexing.method = method
                try:
                    idxr = Indexer.from_parameters(
                        reflections, experiments, params=params
                    )
                    idxr.index()
                except Exception as e:
                    print("Couldn't index using method %s", method)
                    if indexing_error is None:
                        if e is None:
                            e = Exception(f"Couldn't index using method {method}")
                        indexing_error = e
                else:
                    indexing_error = None
                    break
            if indexing_error is not None:
                raise indexing_error

        indexed = idxr.refined_reflections
        experiments = idxr.refined_experiments

        if self.known_crystal_models is not None:

            filtered = flex.reflection_table()
            for idx in set(indexed["miller_index"]):
                sel = indexed["miller_index"] == idx
                if sel.count(True) == 1:
                    filtered.extend(indexed.select(sel))
            print(
                "Filtered duplicate reflections, %d out of %d remaining"
                % (len(filtered), len(indexed))
            )
            indexed = filtered

        return experiments, indexed

    def refine(self, experiments, centroids):
        if self.params.dispatch.refine:
            from dials.algorithms.refinement import RefinerFactory

            refiner = RefinerFactory.from_parameters_data_experiments(
                self.params, centroids, experiments
            )

            refiner.run()
            experiments = refiner.get_experiments()
            predicted = refiner.predict_for_indexed()
            centroids["xyzcal.mm"] = predicted["xyzcal.mm"]
            centroids["entering"] = predicted["entering"]
            centroids = centroids.select(refiner.selection_used_for_refinement())

            # Re-estimate mosaic estimates
            from dials.algorithms.indexing.nave_parameters import NaveParameters

            nv = NaveParameters(
                params=self.params,
                experiments=experiments,
                reflections=centroids,
                refinery=refiner,
                graph_verbose=False,
            )
            nv()
            acceptance_flags_nv = nv.nv_acceptance_flags
            centroids = centroids.select(acceptance_flags_nv)

            # Dump experiments to disk
            if self.params.output.refined_experiments_filename:
                experiments.as_json(self.params.output.refined_experiments_filename)

            if self.params.output.indexed_filename:
                self.save_reflections(centroids, self.params.output.indexed_filename)

        return experiments, centroids

    def integrate(self, experiments, indexed):

        # TODO: Figure out if this is necessary and/or how to do this better
        indexed, _ = self.process_reference(indexed)

        if self.params.integration.integration_only_overrides.trusted_range:
            for detector in experiments.detectors():
                for panel in detector:
                    panel.set_trusted_range(
                        self.params.integration.integration_only_overrides.trusted_range
                    )

        # Get the integrator from the input parameters
        from dials.algorithms.integration.integrator import create_integrator
        from dials.algorithms.profile_model.factory import ProfileModelFactory

        # Compute the profile model
        # Predict the reflections
        # Match the predictions with the reference
        # Create the integrator
        experiments = ProfileModelFactory.create(self.params, experiments, indexed)
        new_experiments = ExperimentList()
        new_reflections = flex.reflection_table()
        for expt_id, expt in enumerate(experiments):
            if (
                    self.params.profile.gaussian_rs.parameters.sigma_b_cutoff is None
                    or expt.profile.sigma_b()
                    < self.params.profile.gaussian_rs.parameters.sigma_b_cutoff
            ):
                refls = indexed.select(indexed["id"] == expt_id)
                refls["id"] = flex.int(len(refls), len(new_experiments))
                # refls.reset_ids()
                del refls.experiment_identifiers()[expt_id]
                refls.experiment_identifiers()[len(new_experiments)] = expt.identifier
                new_reflections.extend(refls)
                new_experiments.append(expt)
            else:
                # TODO: this can be done better, also
                print(
                    "Rejected expt %d with sigma_b %f"
                    % (expt_id, expt.profile.sigma_b())
                )
        experiments = new_experiments
        indexed = new_reflections
        if len(experiments) == 0:
            raise RuntimeError("No experiments after filtering by sigma_b")
        predicted = flex.reflection_table.from_predictions_multi(
            experiments,
            dmin=self.params.prediction.d_min,
            dmax=self.params.prediction.d_max,
            margin=self.params.prediction.margin,
            force_static=self.params.prediction.force_static,
        )
        predicted.match_with_reference(indexed)
        integrator = create_integrator(self.params, experiments, predicted)

        # Integrate the reflections
        integrated = integrator.integrate()

        # correct integrated intensities for absorption correction, if necessary
        for abs_params in self.params.integration.absorption_correction:
            if abs_params.apply:
                if abs_params.algorithm == "fuller_kapton":
                    from dials.algorithms.integration.kapton_correction import (
                        multi_kapton_correction,
                    )
                elif abs_params.algorithm == "kapton_2019":
                    from dials.algorithms.integration.kapton_2019_correction import (
                        multi_kapton_correction,
                    )

                experiments, integrated = multi_kapton_correction(
                    experiments, integrated, abs_params.fuller_kapton, logger=logger
                )()

        if self.params.significance_filter.enable:
            from dials.algorithms.integration.stills_significance_filter import (
                SignificanceFilter,
            )

            sig_filter = SignificanceFilter(self.params)
            filtered_refls = sig_filter(experiments, integrated)
            accepted_expts = ExperimentList()
            accepted_refls = flex.reflection_table()

            for expt_id, expt in enumerate(experiments):
                refls = filtered_refls.select(filtered_refls["id"] == expt_id)
                if len(refls) > 0:
                    accepted_expts.append(expt)
                    refls["id"] = flex.int(len(refls), len(accepted_expts) - 1)
                    accepted_refls.extend(refls)
                else:
                    print(
                        "Removed experiment %d which has no reflections left after applying significance filter",
                        expt_id,
                    )

            if len(accepted_refls) == 0:
                raise Sorry("No reflections left after applying significance filter")
            experiments = accepted_expts
            integrated = accepted_refls

        # Delete the shoeboxes used for intermediate calculations, if requested
        if self.params.integration.debug.delete_shoeboxes and "shoebox" in integrated:
            del integrated["shoebox"]

        # Dump experiments to disk
        if self.params.output.integrated_experiments_filename:
            experiments.as_json(self.params.output.integrated_experiments_filename)

        if self.params.output.integrated_filename:
            # Save the reflections
            self.save_reflections(
                integrated, self.params.output.integrated_filename
            )

        self.write_integration_pickles(integrated, experiments)

        # TODO: Figure out what this is
        from dials.algorithms.indexing.stills_indexer import (
            calc_2D_rmsd_and_displacements,
        )

        rmsd_indexed, _ = calc_2D_rmsd_and_displacements(indexed)
        log_str = f"RMSD indexed (px): {rmsd_indexed:f}\n"
        for i in range(6):
            bright_integrated = integrated.select(
                (
                        integrated["intensity.sum.value"]
                        / flex.sqrt(integrated["intensity.sum.variance"])
                )
                >= i
            )
            if len(bright_integrated) > 0:
                rmsd_integrated, _ = calc_2D_rmsd_and_displacements(bright_integrated)
            else:
                rmsd_integrated = 0
            log_str += (
                    "N reflections integrated at I/sigI >= %d: % 4d, RMSD (px): %f\n"
                    % (i, len(bright_integrated), rmsd_integrated)
            )

        for crystal_model in experiments.crystals():
            if hasattr(crystal_model, "get_domain_size_ang"):
                log_str += ". Final ML model: domain size angstroms: {:f}, half mosaicity degrees: {:f}".format(
                    crystal_model.get_domain_size_ang(),
                    crystal_model.get_half_mosaicity_deg(),
                )

        print(log_str)
        return integrated

    def construct_frame(self, integrated, exeriments):
        # Construct frame
        from xfel.command_line.frame_extractor import ConstructFrame

        self.frame = ConstructFrame(integrated, experiments[0]).make_frame()
        self.frame["pixel_size"] = experiments[0].detector[0].get_pixel_size()[0]

    def write_integration_pickles(self, integrated, experiments):
        if self.write_pickle:
            from libtbx import easy_pickle

            if not hasattr(self, frame):
                self.construct_frame(integrated, experiments)
            easy_pickle.dump(self.params.output.integration_pickle, self.frame)

    def process_reference(self, reference):
        """Load the reference spots."""
        if reference is None:
            return None, None
        st = time.time()
        assert "miller_index" in reference
        assert "id" in reference
        mask = reference.get_flags(reference.flags.indexed)
        rubbish = reference.select(~mask)
        if mask.count(False) > 0:
            reference.del_selected(~mask)
        if len(reference) == 0:
            raise Sorry(
                """ Invalid input for reference reflections. Expected > %d indexed 
                spots, got %d """ % (0, len(reference))
            )
        mask = reference["miller_index"] == (0, 0, 0)
        if mask.count(True) > 0:
            rubbish.extend(reference.select(mask))
            reference.del_selected(mask)
        mask = reference["id"] < 0
        if mask.count(True) > 0:
            raise Sorry(
                """ Invalid input for reference reflections. %d reference spots have 
                an invalid experiment id """
                % mask.count(True)
            )
        return reference, rubbish

    def save_reflections(self, reflections, filename):
        """Save the reflections to file."""
        reflections.as_file(filename)

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
            with utils.Capturing() as junk_output:
                img_object = self.integrator.run(img_object)

        # Update main log
        if hasattr(self.info, "logfile"):
            main_log_entry = "\n".join(img_object.log_info)
            utils.main_log(self.info.logfile, main_log_entry)
            utils.main_log(self.info.logfile, "\n{:-^100}\n".format(""))

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
        from iota.analysis.iota_analysis import Analyzer

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
        from iota.init.image_import import ImageImporter as Importer
        from iota.processing.processing import IOTAImageProcessor as Integrator

        # Initialize processing parameters
        from iota.init.iota_init import initialize_processing

        cls.info, cls.params = initialize_processing(paramfile, run_no)

        cls.importer = Importer(info=cls.info)
        cls.integrator = Integrator(iparams=cls.params)

        return cls(*args, **kwargs)

    @classmethod
    def for_existing_run(cls, info, *args, **kwargs):

        from iota.init.image_import import ImageImporter as Importer
        from iota.processing.processing import IOTAImageProcessor as Integrator

        # Initialize processing parameters
        from iota.init.iota_init import resume_processing

        cls.info, cls.params = resume_processing(info)

        cls.importer = Importer(info=cls.info)
        cls.integrator = Integrator(iparams=cls.params)

        return cls(*args, **kwargs)

    @classmethod
    def for_single_image(
        cls, info, params, action_code="spotfinding", verbose=False, *args, **kwargs
    ):

        from iota.init.image_import import ImageImporter
        from iota.processing.iota_processing import IOTAImageProcessor

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
