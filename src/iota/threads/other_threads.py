from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 02/08/2021
Last Changed: 02/08/2021
Description : Assorted Threads and PostEvents
"""

import multiprocessing
import os
from threading import Thread

import wx
from dxtbx.model.experiment_list import ExperimentListFactory
from libtbx import easy_run
from libtbx.easy_mp import parallel_map

tp_EVT_SPFDONE = wx.NewEventType()
EVT_SPFDONE = wx.PyEventBinder(tp_EVT_SPFDONE, 1)
tp_EVT_SPFALLDONE = wx.NewEventType()
EVT_SPFALLDONE = wx.PyEventBinder(tp_EVT_SPFALLDONE, 1)
tp_EVT_SPFTERM = wx.NewEventType()
EVT_SPFTERM = wx.PyEventBinder(tp_EVT_SPFTERM)


class SpotFinderAllDone(wx.PyCommandEvent):
    """Send event when finished all cycles."""

    def __init__(self, etype, eid, info=None):
        wx.PyCommandEvent.__init__(self, etype, eid)
        self.info = info

    def GetValue(self):
        return self.info


class SpotFinderOneDone(wx.PyCommandEvent):
    """Send event when finished all cycles."""

    def __init__(self, etype, eid, info=None):
        wx.PyCommandEvent.__init__(self, etype, eid)
        self.info = info

    def GetValue(self):
        return self.info


class SpotFinderTerminated(wx.PyCommandEvent):
    """Send event when spotfinder terminated."""

    def __init__(self, etype, eid):
        wx.PyCommandEvent.__init__(self, etype, eid)

    def GetValue(self):
        return None


class SpotFinderDIALSThread:
    def __init__(
        self, parent, processor, term_file, run_indexing=False, run_integration=False
    ):
        self.meta_parent = parent.parent
        self.processor = processor
        self.term_file = term_file
        self.run_indexing = run_indexing
        self.run_integration = run_integration

    def run(self, idx, img):
        if self.meta_parent.terminated:
            raise IOTATermination("IOTA_TRACKER: SPF Termination signal received!")
        else:
            with Capturing() as junk_output:
                fail = False
                sg = None
                uc = None
                try:
                    experiments = ExperimentListFactory.from_filenames([img])[0]
                    observed = self.processor.find_spots(experiments=experiments)
                except Exception:
                    fail = True
                    observed = []
                    pass

                # TODO: Indexing / lattice determination very slow (how to speed up?)
                if self.run_indexing:
                    if not fail:
                        try:
                            experiments, indexed = self.processor.index(
                                experiments=experiments, reflections=observed
                            )
                        except Exception:
                            fail = True
                            pass

                    if not fail:
                        try:
                            solution = self.processor.refine_bravais_settings(
                                reflections=indexed, experiments=experiments
                            )

                            # Only reindex if higher-symmetry solution found
                            if solution is not None:
                                experiments, indexed = self.processor.reindex(
                                    reflections=indexed,
                                    experiments=experiments,
                                    solution=solution,
                                )
                            lat = experiments[0].crystal.get_space_group().info()
                            sg = str(lat).replace(" ", "")
                        except Exception:
                            fail = True
                            pass

                    if not fail:
                        unit_cell = experiments[0].crystal.get_unit_cell().parameters()
                        uc = " ".join(["{:.4f}".format(i) for i in unit_cell])

                    if self.run_integration:
                        if not fail:
                            try:
                                # Run refinement
                                experiments, indexed = self.processor.refine(
                                    experiments=experiments, centroids=indexed
                                )
                            except Exception:
                                fail = True
                                pass

                        if not fail:
                            try:
                                print(experiments)
                                print(indexed)
                                integrated = self.processor.integrate(
                                    experiments=experiments, indexed=indexed
                                )
                            except Exception:
                                pass

            return [idx, int(len(observed)), img, sg, uc]


class SpotFinderMosflmThread:
    def __init__(self, parent, term_file):
        self.meta_parent = parent.parent
        self.term_file = term_file

    def run(self, idx, img):
        if os.path.isfile(self.term_file):
            raise IOTATermination("IOTA_TRACKER: Termination signal received!")
        else:
            # First, parse filepath to create Mosflm template
            directory = os.path.dirname(img)
            filepath = os.path.basename(img).split(".")
            fname = filepath[0]
            extension = filepath[1]
            if "_" in fname:
                suffix = fname.split("_")[-1]
            elif "-" in fname:
                suffix = fname.split("-")[-1]
            elif "." in fname:
                suffix = fname.split(".")[-1]
            else:
                suffix = fname
            img_number = int("".join(n if n.isdigit() else "" for n in suffix))
            prefix = fname.replace(suffix, "")
            n_suffix = "".join("#" if c.isdigit() else c for c in suffix)
            template = "{}{}.{}".format(prefix, n_suffix, extension)

            # Create autoindex.com w/ Mosflm script
            # Write to temporary file and change permissions to run
            autoindex = [
                "#! /bin/tcsh -fe",
                "ipmosflm << eof-ipmosflm".format(fname),
                "NEWMATRIX {0}.mat".format(fname),
                "DIRECTORY {}".format(directory),
                "TEMPLATE {}".format(template),
                "AUTOINDEX DPS THRESH 0.1 IMAGE {} PHI 0 0.01".format(img_number),
                "GO",
                "eof-ipmosflm",
            ]
            autoindex_string = "\n".join(autoindex)
            autoindex_filename = "autoindex_{}.com".format(idx)

            with open(autoindex_filename, "w") as af:
                af.write(autoindex_string)
            os.chmod(autoindex_filename, 0o755)

            # Run Mosflm autoindexing
            command = "./{}".format(autoindex_filename)
            out = easy_run.fully_buffered(command, join_stdout_stderr=True)

            # Scrub text output
            final_spots = [
                l for l in out.stdout_lines if "spots written for image" in l
            ]
            final_cell_line = [l for l in out.stdout_lines if "Final cell" in l]
            final_sg_line = [l for l in out.stdout_lines if "space group" in l]

            if final_spots:
                spots = final_spots[0].rsplit()[0]
            else:
                spots = 0
            if final_cell_line:
                cell = (
                    final_cell_line[0]
                    .replace("Final cell (after refinement) is", "")
                    .rsplit()
                )
            else:
                cell = None
            if final_sg_line:
                sg = final_sg_line[0].rsplit()[6]
            else:
                sg = None

            # Temp file cleanup
            try:
                os.remove("{}.mat".format(fname))
            except Exception:
                pass
            try:
                os.remove("{}.spt".format(prefix[:-1]))
            except Exception:
                pass
            try:
                os.remove("SUMMARY")
            except Exception:
                pass
            try:
                os.remove(autoindex_filename)
            except Exception:
                pass

            return [idx, spots, img, sg, cell]


class SpotFinderThread(Thread):
    """Basic spotfinder (with defaults) that could be used to rapidly analyze
    images as they are collected."""

    def __init__(
        self,
        parent,
        data_list=None,
        term_file=None,
        proc_params=None,
        backend="dials",
        n_proc=0,
        run_indexing=False,
        run_integration=False,
    ):
        Thread.__init__(self)
        self.parent = parent
        self.data_list = data_list
        self.term_file = term_file
        self.terminated = False
        self.backend = backend
        self.run_indexing = run_indexing
        self.run_integration = run_integration
        if n_proc > 0:
            self.n_proc = n_proc
        else:
            self.n_proc = multiprocessing.cpu_count() - 2

        if self.backend == "dials":
            # Modify default DIALS parameters
            # These parameters will be set no matter what
            proc_params.output.experiments_filename = None
            proc_params.output.indexed_filename = None
            proc_params.output.strong_filename = None
            proc_params.output.refined_experiments_filename = None
            proc_params.output.integrated_filename = None
            proc_params.output.integrated_experiments_filename = None
            proc_params.output.profile_filename = None
            proc_params.output.integration_pickle = None

            from iota.processing.processing import IOTAImageProcessor

            self.processor = IOTAImageProcessor(phil=proc_params)

    def run(self):
        try:
            parallel_map(
                iterable=self.data_list,
                func=self.spf_wrapper,
                callback=self.callback,
                processes=self.n_proc,
            )
        except IOTATermination as e:
            self.terminated = True
            print(e)

        # Signal that this batch is finished
        try:
            if self.terminated:
                print("RUN TERMINATED!")
                evt = SpotFinderTerminated(tp_EVT_SPFTERM, -1)
                wx.PostEvent(self.parent, evt)

            wx.CallAfter(self.parent.onSpfAllDone, self.data_list)

            # info = self.data_list
            # evt = SpotFinderAllDone(tp_EVT_SPFALLDONE, -1, info=info)
            # wx.PostEvent(self.parent, evt)
            return
        except TypeError as e:
            print(e)
            return

    def spf_wrapper(self, img):
        try:
            if os.path.isfile(img):
                if self.backend == "dials":
                    spf_worker = SpotFinderDIALSThread(
                        self,
                        processor=self.processor,
                        term_file=self.term_file,
                        run_indexing=self.run_indexing,
                        run_integration=self.run_integration,
                    )
                    result = spf_worker.run(idx=int(self.data_list.index(img)), img=img)
                elif self.backend == "mosflm":
                    spf_worker = SpotFinderMosflmThread(self, self.term_file)
                    result = spf_worker.run(idx=int(self.data_list.index(img)), img=img)
                else:
                    result = [int(self.data_list.index(img)), 0, img, None, None]
                return result
            else:
                return [int(self.data_list.index(img)), 0, img, None, None]
        except IOTATermination as e:
            raise e

    def callback(self, info):
        try:
            wx.CallAfter(self.parent.onSpfOneDone, info)
            # evt = SpotFinderOneDone(tp_EVT_SPFDONE, -1, info=info)
            # wx.PostEvent(self.parent.parent, evt)
        except TypeError:
            pass

    # def terminate_thread(self):
    #   raise IOTATermination('IOTA_TRACKER: SPF THREAD Terminated!')


class InterceptorFileThread(Thread):
    def __init__(self, parent, results_file, reorder=False):
        Thread.__init__(self)
        self.parent = parent
        self.results_file = results_file
        self.reorder = reorder

        self.bookmark = 0
        self.msg = ""
        self.spotfinding_info = []
        self.cluster_info = None

        self.prc_timer = wx.Timer()
        self.cls_timer = wx.Timer()

        # Bindings
        self.prc_timer.Bind(wx.EVT_TIMER, self.onProcTimer)
        self.cls_timer.Bind(wx.EVT_TIMER, self.onClusterTimer)

    def run(self):
        # self.timer.Start(1000)
        pass

    def onProcTimer(self, e):
        if os.path.isfile(self.results_file):
            with open(self.results_file, "r") as rf:
                rf.seek(self.bookmark)
                split_info = [i.replace("\n", "").split(" ") for i in rf.readlines()]
                self.bookmark = rf.tell()

            if self.reorder:
                idx_offset = len(self.spotfinding_info)
                new_info = [
                    [
                        split_info.index(i) + idx_offset,
                        int(i[1]),
                        i[2],
                        i[3],
                        tuple(i[4:10]),
                    ]
                    if len(i) > 5
                    else [
                        split_info.index(i) + idx_offset,
                        int(i[1]),
                        i[2],
                        util.makenone(i[3]),
                        util.makenone(i[4]),
                    ]
                    for i in split_info
                ]
            else:
                new_info = [
                    [int(i[0]), int(i[1]), i[2], i[3], tuple(i[4:10])]
                    if len(i) > 5
                    else [
                        int(i[0]),
                        int(i[1]),
                        i[2],
                        util.makenone(i[3]),
                        util.makenone(i[4]),
                    ]
                    for i in split_info
                ]

            if len(new_info) > 0:
                self.spotfinding_info.extend(new_info)

                if len(self.spotfinding_info) > 0:
                    self.msg = "Tracking new images in {} ...".format(self.results_file)
            else:
                self.msg = "Waiting for new images in {} ...".format(self.results_file)

        else:
            self.msg = "Waiting for new run to initiate..."

        info = [self.msg, self.spotfinding_info, self.cluster_info]
        evt = SpotFinderOneDone(tp_EVT_SPFDONE, -1, info=info)
        wx.PostEvent(self.parent, evt)

    def onClusterTimer(self, e):
        input = []
        for item in self.spotfinding_info:
            if item[4] is not None:
                try:
                    if type(item[4]) in (tuple, list):
                        uc = item[4]
                    else:
                        uc = item[4].rsplit()
                    info_line = [float(i) for i in uc]
                    info_line.append(item[3])
                    input.append(info_line)
                except ValueError as e:
                    print("CLUSTER ERROR: ", e)
                    pass

        if len(input) > 0:
            self.running_clustering = True
            cluster_thread = ClusterWorkThread(self)
            self.cluster_info = cluster_thread.run(iterable=input)

    def terminate_thread(self):
        raise IOTATermination("IOTA_TRACKER: Termination signal received!")


class InterceptorThread(Thread):
    """Thread for the full Interceptor image processing process; will also
    house the processing timer, which will update the UI front end and initiate
    plotting."""

    def __init__(
        self,
        parent,
        data_folder=None,
        term_file=None,
        proc_params=None,
        backend="dials",
        n_proc=0,
        min_back=None,
        run_indexing=False,
        run_integration=False,
    ):
        Thread.__init__(self)
        self.parent = parent
        self.data_folder = data_folder
        self.term_file = term_file
        self.terminated = False
        self.backend = backend
        self.run_indexing = run_indexing
        self.run_integration = run_integration
        self.min_back = min_back
        self.submit_new_images = True

        self.spotfinding_info = []
        self.cluster_info = None
        self.msg = None
        self.done_list = []
        self.data_list = []
        self.new_data = []

        self.spf_thread = None

        if n_proc > 0:
            self.n_proc = n_proc
        else:
            self.n_proc = multiprocessing.cpu_count() - 2

        if self.backend == "dials":
            # Modify default DIALS parameters
            # These parameters will be set no matter what
            proc_params.output.experiments_filename = None
            proc_params.output.indexed_filename = None
            proc_params.output.strong_filename = None
            proc_params.output.refined_experiments_filename = None
            proc_params.output.integrated_filename = None
            proc_params.output.integrated_experiments_filename = None
            proc_params.output.profile_filename = None
            proc_params.output.integration_pickle = None

            self.proc_params = proc_params
            self.prc_timer = wx.Timer()
        self.cls_timer = wx.Timer()

        # Bindings
        self.prc_timer.Bind(wx.EVT_TIMER, self.onProcTimer)
        self.cls_timer.Bind(wx.EVT_TIMER, self.onClusterTimer)

    def run(self):
        pass

    def onProcTimer(self, e):
        """Main timer (1 sec) will send data to UI, find new images, and submit
        new processing run."""

        # Send current data to UI
        info = [self.msg, self.spotfinding_info, self.cluster_info]
        evt = SpotFinderOneDone(tp_EVT_SPFDONE, -1, info=info)
        wx.PostEvent(self.parent, evt)

        # Find new images
        if self.data_list:
            last_file = self.data_list[-1]
        else:
            last_file = None
        self.find_new_images(last_file=last_file, min_back=self.min_back)

        if self.spf_thread is not None:
            if not self.spf_thread.isAlive():
                self.submit_new_images = True

        # Submit new images (if found)
        if self.submit_new_images and len(self.new_data) > 0:
            self.submit_new_images = False
            self.run_processing()

    def onClusterTimer(self, e):
        input = []
        if len(self.spotfinding_info) > 0:
            for item in self.spotfinding_info:
                if item[4] is not None:
                    try:
                        if type(item[4]) in (tuple, list):
                            uc = item[4]
                        else:
                            uc = item[4].rsplit()
                        info_line = [float(i) for i in uc]
                        info_line.append(item[3])
                        input.append(info_line)
                    except ValueError as e:
                        print("CLUSTER ERROR: ", e)
                        pass

            if len(input) > 0:
                self.running_clustering = True
                cluster_thread = ClusterWorkThread(self)
                self.cluster_info = cluster_thread.run(iterable=input)

    def find_new_images(self, min_back=None, last_file=None):
        found_files, _ = ginp.make_input_list(
            [self.data_folder],
            filter_results=True,
            filter_type="image",
            last=last_file,
            min_back=min_back,
            expand_multiple=True,
        )

        # Sometimes duplicate files are found anyway; clean that up
        found_files = list(set(found_files) - set(self.data_list))

        # Add new files to the data list & clean up
        self.new_data.extend(found_files)
        self.new_data = sorted(self.new_data, key=lambda i: i)
        self.data_list.extend(self.new_data)

    def run_processing(self):
        self.spf_thread = SpotFinderThread(
            self,
            data_list=self.new_data,
            term_file=self.term_file,
            proc_params=self.proc_params,
            backend=self.backend,
            n_proc=self.n_proc,
            run_indexing=self.run_indexing,
            run_integration=self.run_integration,
        )
        self.new_data = []
        self.spf_thread.start()

    def onSpfOneDone(self, info):
        info[0] = int(info[0]) + len(self.done_list)
        self.spotfinding_info.append(info)

    def onSpfAllDone(self, done_list):
        self.done_list.extend(done_list)

    def terminate_thread(self):
        self.terminated = True
