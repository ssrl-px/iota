from __future__ import absolute_import, division, print_function

import iota.threads.iota_threads

"""
Author      : Lyubimov, A.Y.
Created     : 02/08/2021
Last Changed: 02/08/2021
Description : IOTA processing info object
"""

import collections
import json
import os

from libtbx import easy_pickle as ep
from past.types import basestring
from iota.utils import utils


class ProcInfo(object):
    """Container for all the processing info.

    Updated by Object Sentinel during processing. Stored as JSON dict
    and can be used to recover a previous run.
    """

    def __init__(self, info_dict=None):
        """Constructor.

        :param dict: dictionary of attributes, likely from JSON file
        """

        if info_dict:

            # Convert all unicode values to strings
            info_dict = self._make_serializable(info_dict)

            # update with values from dictionary
            self.update(info_dict)

    def update(self, info_dict=None, **kwargs):
        """Add / overwrite attributes from dictionary and/or set of kwargs."""
        for dictionary in (info_dict, kwargs):
            if dictionary:
                for key, value in dictionary.items():
                    setattr(self, key, value)

    def _select_image_range(self, full_list, range_str):
        """Selects a range of images (can be complex)"""
        img_range_elements = range_str.split(",")
        img_list = []
        for n in img_range_elements:
            if "-" in n:
                img_limits = [int(i) for i in n.split("-")]
                start = min(img_limits) - 1
                end = max(img_limits)
                if start <= len(full_list) and end <= len(full_list):
                    img_list.extend(full_list[start:end])
            else:
                if int(n) <= len(full_list):
                    img_list.append(full_list[int(n)])

        if len(img_list) > 0:
            return img_list
        else:
            return full_list

    def _select_random_subset(self, full_list, number=0):
        """Selects random subset of input entries."""
        import random

        random_inp_list = []
        if number == 0:
            if len(full_list) <= 5:
                number = len(full_list)
            elif len(full_list) <= 50:
                number = 5
            else:
                number = int(len(full_list) * 0.1)

        for i in range(number):
            random_number = random.randrange(0, len(full_list))
            if full_list[random_number] in random_inp_list:
                while full_list[random_number] in random_inp_list:
                    random_number = random.randrange(0, len(full_list))
                random_inp_list.append(full_list[random_number])
            else:
                random_inp_list.append(full_list[random_number])

        return random_inp_list

    def flatten_input(self, **kw):
        """Generates a flat list of absolute imagefile paths.

        :param kw: Typical kwargs may include a param file, path to images,
        whether to select a random subset of imagefiles and/or a range, etc.
        :return: imagefile path iterator
        """

        if "params" in kw:
            prm = kw["params"]
        elif hasattr(self, "params"):
            prm = self.params
        else:
            prm = None

        input_list = None
        if hasattr(self, "input_list_file"):
            input_filepath = self.input_list_file
        elif "filepath" in kw:
            input_filepath = kw["filepath"]
        else:
            input_filepath = None

        if input_filepath:
            inputs = [input_filepath]
        else:
            if "input" in kw:
                inputs = kw["input"]
            elif prm:
                inputs = [i for i in prm.input if i is not None]
            else:
                inputs = None

        if inputs:
            input_list, _ = iota.threads.iota_threads.ginp.make_input_list(
                inputs, filter_results=True, filter_type="image", expand_multiple=True
            )

        # Select random subsets and/or ranges of images
        if prm and input_list:
            if prm.data_selection.image_range.flag_on:
                input_list = self._select_image_range(
                    input_list, prm.data_selection.image_range.range
                )
            if prm.data_selection.random_sample.flag_on:
                input_list = self._select_random_subset(
                    input_list, prm.data_selection.random_sample.number
                )

        return sorted(input_list)

    def generate_input_list(self, **kw):
        assert not hasattr(self, "input_list")
        self.input_list = self.flatten_input(**kw)
        self.unprocessed = []
        for i in self.input_list:
            idx = self.input_list.index(i) + 1
            if isinstance(i, str):
                path = i
                img_index = 0
            else:
                path, img_index = i
            self.unprocessed.append([idx, path, img_index])
        self.n_input_images = len(self.unprocessed)

    def update_input_list(self, new_input=None):
        assert hasattr(self, "input_list") and hasattr(self, "categories")
        if new_input:
            self.input_list += new_input
            self.unprocessed = []
            for i in new_input:
                idx = new_input.index(i) + self.n_input_images
                if isinstance(i, str):
                    path = i
                    img_index = 0
                else:
                    path, img_index = i
                self.unprocessed.append([idx, path, img_index])
            self.categories["not_processed"][0].extend(self.unprocessed)
            self.categories["total"][0].extend(self.unprocessed)
            self.n_input_images += len(self.unprocessed)

    def reset_input_list(self):
        assert hasattr(self, "input_list") and hasattr(self, "categories")
        self.input_list = [(str(i[1]), i[2]) for i in self.unprocessed]

    def get_finished_objects(self):
        if hasattr(self, "finished_objects") and self.finished_objects:
            return (ep.load(o) for o in self.finished_objects)

    def get_finished_objects_from_filelist(self, filelist):
        assert filelist
        return (ep.load(o) for o in filelist)

    def get_finished_objects_from_file(self):
        if hasattr(self, "obj_list_file") and os.path.isfile(self.obj_list_file):
            with open(self.obj_list_file, "r") as objf:
                objf.seek(self.bookmark)
                obj_paths = [i.rstrip("\n") for i in objf.readlines()]
                self.bookmark = objf.tell()
            if obj_paths:
                self.finished_objects.extend(obj_paths)
                return (ep.load(o) for o in obj_paths)
            else:
                return None

    def get_final_objects(self):
        if hasattr(self, "final_objects") and self.final_objects:
            return (ep.load(o) for o in self.final_objects)

    # def get_observations(self, to_p1=True):
    #   if hasattr(self, 'final_objects') and self.final_objects:
    #
    #     # Read final objects into a generator
    #     fin = (ep.load(o) for o in self.final_objects)
    #
    #     # Extract miller_index objects from final_objects or integrated pickles
    #     all_obs = []
    #     for obj in fin:
    #       if 'observations' in obj.final:
    #         obs = obj.final['observations'].as_non_anomalous_array()
    #       else:
    #         obs = ep.load(obj.final['final'])['observations'][0].as_non_anomalous_array()
    #       all_obs.append(obs)
    #
    #     # Combine miller_index objects into a single miller_index object
    #     with utils.Capturing():
    #       observations = None
    #       for o in all_obs[1:]:
    #         if to_p1:
    #           o = o.change_symmetry('P1')
    #         if observations:
    #           observations = observations.concatenate(o, assert_is_similar_symmetry=False)
    #         else:
    #           observations = o
    #     return observations

    # def update_indices(self, filepath=None, obs=None):
    #   if not filepath:
    #     filepath = self.idx_file
    #   if not (hasattr(self, 'merged_indices')):
    #     self.merged_indices = {}
    #   if not obs:
    #     obs = ep.load(filepath)
    #
    #   with utils.Capturing():
    #     p1_mrg = obs.change_symmetry('P1').merge_equivalents()
    #     p1_red = p1_mrg.redundancies()
    #   for i in p1_red:
    #     hkl = ' '.join([str(j) for j in i[0]])
    #     red = i[1]
    #     if hkl in self.merged_indices:
    #       self.merged_indices[hkl] += red
    #     else:
    #       self.merged_indices[hkl] = red
    #
    #   with open(os.path.join(self.int_base, 'merged.json'), 'w') as mjf:
    #     json.dump(self.merged_indices, mjf)

    # dict to list:
    # idx_list = [tuple([tuple(int(i) for i in k.split(' ')), v])
    #             for k, v in self.merged_indices.items()]

    def get_hkl_slice(self, sg="P1", axis="l"):

        try:
            all_obs = ep.load(self.idx_file)
        except IOError:
            return None

        with utils.Capturing():
            ext_sg = str(all_obs.space_group_info()).replace(" ", "").lower()
            sg = sg.replace(" ", "").lower()
            if ext_sg != sg:
                all_obs = all_obs.change_symmetry(sg, merge_non_unique=False)

        # Recalculate redundancies and slices from updated indices
        red = all_obs.merge_equivalents().redundancies()
        return red.slice(axis=axis, slice_start=0, slice_end=0)

    def export_json(self, **kw):
        """Export contents as JSON dict."""

        if "filepath" in kw:
            json_file = kw["filepath"]
        elif hasattr(self, "info_file"):
            json_file = self.info_file
        else:
            if hasattr(self, "int_base"):
                int_base = self.int_base
            else:
                int_base = os.path.abspath(os.curdir)
            json_file = os.path.join(int_base, "proc.info")

        try:
            with open(json_file, "w") as jf:
                json.dump(self.__dict__, jf)
        except TypeError:
            # pick up non-serializable objects when json.dump fails; putting this
            # into a try block because inefficient with large INFO objects, esp. in
            # Python2
            try:
                with open(json_file, "w") as jf:
                    json.dump(self._make_serializable(self.__dict__), jf)
            except TypeError as e:
                raise Exception("IOTA JSON ERROR: {}".format(e))

    def _make_serializable(self, info_dict):
        if isinstance(info_dict, basestring):
            return str(info_dict)
        elif isinstance(info_dict, collections.Mapping):
            return dict(map(self._make_serializable, info_dict.items()))
        elif isinstance(info_dict, collections.Iterable):
            return type(info_dict)(map(self._make_serializable, info_dict))
        else:
            if type(info_dict).__module__ == "numpy":
                info_dict = info_dict.item()
            return info_dict

    @classmethod
    def from_json(cls, filepath, **kwargs):
        """Generate INFO object from a JSON file."""

        # Check for file
        if not os.path.isfile(filepath):
            return None

        # Read in JSON file
        with open(filepath, "r") as json_file:
            info_dict = json.load(json_file)

        # Allow to override param values with in-code args, etc.
        info_dict.update(kwargs)

        return cls(info_dict)

    @classmethod
    def from_pickle(cls, filepath):
        """To recover an old pickled info object."""
        pass

    @classmethod
    def from_dict(cls, info_dict):
        return cls(info_dict)

    @classmethod
    def from_args(cls, **kwargs):
        return cls(kwargs)

    @classmethod
    def from_folder(cls, path):
        """Generate INFO object from an integration folder.

        :param path: path to folder with integration results
        :return: ProcInfo class generated with attributes
        """

        # Check for folder
        if not os.path.isdir(path):
            return None

        # Check for proc.info file
        info_path = os.path.isfile(os.path.join(path, "proc.info"))
        if info_path:
            try:
                with open(os.path.join(path, "proc.info"), "r") as json_file:
                    info_dict = json.load(json_file)
                    return cls(info_dict)
            except Exception:
                pass

        # If file not there, reconstruct from contents of folder
        # Create info dictionary
        obj_base = os.path.join(path, "image_objects")
        fin_base = os.path.join(path, "final")
        log_base = os.path.join(path, "logs")
        viz_base = os.path.join(path, "visualization")
        logfile = os.path.join(path, "iota.log")
        info_dict = dict(
            int_base=path,
            obj_base=obj_base,
            fin_base=fin_base,
            log_base=log_base,
            viz_base=viz_base,
            logfile=logfile,
        )

        # Logfile is pretty much the most key element of an IOTA run
        if not os.path.isfile(logfile):
            return None

        # Look for the IOTA paramfile
        prm_list = iota.threads.iota_threads.ginp.get_input_from_folder(path=path, ext_only="param")
        if prm_list:  # Read from paramfile
            with open(prm_list[0], "r") as prmf:
                info_dict["iota_phil"] = prmf.read()
        else:  # Last ditch: extract from log
            with open(logfile, "r") as logf:
                lines = []
                while True:
                    line = next(logf)
                    if "-----" in line:
                        break
                while True:
                    line = next(logf)
                    if "-----" in line:
                        break
                    else:
                        lines.append(line)
            if lines:
                info_dict["iota_phil"] = "".join(lines)

        # Look for the target PHIL file
        target_file = os.path.join(path, "target.phil")
        if os.path.isfile(target_file):  # Read from target file
            with open(target_file, "r") as tarf:
                info_dict["target_phil"] = tarf.read()
        else:  # Last ditch: extract from log
            with open(logfile, "r") as logf:
                lines = []
                while True:
                    line = next(logf)
                    if "BACKEND SETTINGS" in line:
                        break
                while True:
                    line = next(logf)
                    if "-----" in line:
                        break
                    else:
                        lines.append(line)
            if lines:
                info_dict["target_phil"] = "".join(lines)

        # Generate object list
        if os.path.isdir(obj_base):
            info_dict["finished_objects"] = iota.threads.iota_threads.ginp.get_input_from_folder(
                path, ext_only="int"
            )

        return cls(info_dict)

# --> end
