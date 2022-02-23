from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 02/08/2021
Last Changed: 02/08/2021
Description : Analysis Threads and PostEvents
"""

import os
import shutil
from threading import Thread

import wx
from cctbx import crystal
from cctbx.sgtbx import lattice_symmetry
from cctbx.uctbx import unit_cell
from iota.threads.other_threads import SpotFinderOneDone
from libtbx import easy_pickle as ep
from xfel.clustering.cluster import Cluster

tp_EVT_PRIMEDONE = wx.NewEventType()
EVT_PRIMEDONE = wx.PyEventBinder(tp_EVT_PRIMEDONE, 1)


class PRIMEAllDone(wx.PyCommandEvent):
    """Send event when finished all cycles."""

    def __init__(self, etype, eid, info=None):
        wx.PyCommandEvent.__init__(self, etype, eid)
        self.info = info

    def GetValue(self):
        return self.info


class PRIMEThread(Thread):
    """Thread for running all PRIME calculations; will prepare info for
    plotting in main GUI thread."""

    def __init__(self, parent, info, params, best_pg=None, best_uc=None):
        Thread.__init__(self, name="live_prime")
        self.parent = parent
        self.params = params
        self.info = info
        self.best_pg = best_pg
        self.best_uc = best_uc

    def prepare_PRIME_input(self):
        """Prepare the list of integrated pickles as well as pertinent
        parameters; create a PRIME input file."""

        # Check if any pg/uc information is available; pg/uc info from UI
        # supercedes pg/uc from Cluster

        if self.best_pg is None:
            self.best_pg = self.info.best_pg

        if self.best_uc is None:
            self.best_uc = self.info.best_uc
        else:
            uc_params = str(self.best_uc).rsplit()
            self.best_uc = [float(i) for i in uc_params]

        # Only run PRIME if both pg and uc are provided
        if self.best_pg and self.best_uc:
            from iota.analysis.iota_analysis import Analyzer

            # Create a file with list of integrated pickles (overwrite old file)
            int_pickles_file = os.path.join(self.info.int_base, "int_pickles.lst")
            with open(int_pickles_file, "w") as ff:
                ff.write("\n".join(self.info.categories["integrated"][0]))

            # Create PRIME input file
            analyzer = Analyzer(info=self.info, params=self.params)
            analyzer.prime_data_path = int_pickles_file
            analyzer.best_pg = self.best_pg
            analyzer.best_uc = self.best_uc

            prime_phil = analyzer.make_prime_input(
                filename="live_prime.phil", run_zero=True
            )
            self.pparams = prime_phil.extract()

            # Modify specific options based in IOTA settings
            # Queue options
            if self.params.mp.method == "lsf" and self.params.mp.queue is not None:
                self.pparams.queue.mode = "bsub"
                self.pparams.queue.qname = self.params.mp.queue

            # Number of processors (automatically, 1/2 of IOTA procs)
            self.pparams.n_processors = int(self.params.mp.n_processors / 2)

            # Generate command args
            cmd_args_list = [
                "n_postref_cycle=0",
                "queue.mode={}".format(self.pparams.queue.mode),
                "queue.qname={}".format(self.pparams.queue.qname),
                "n_processors={}".format(self.pparams.n_processors),
            ]
            if self.pparams.queue.mode == "bsub":
                cmd_args_list.append("timeout_seconds=120")
            cmd_args = " ".join(cmd_args_list)

            return cmd_args
        else:
            return None

    def get_prime_stats(self):
        stats_folder = os.path.join(self.pparams.run_no, "stats")
        prime_info = None
        if os.path.isdir(stats_folder):
            stat_files = [
                os.path.join(stats_folder, i)
                for i in os.listdir(stats_folder)
                if i.endswith("stat")
            ]
            if stat_files:
                assert len(stat_files) == 1
                stat_file = stat_files[0]
                if os.path.isfile(stat_file):
                    prime_info = ep.load(stat_file)
                    live_prime_info_file = os.path.join(
                        self.info.int_base, "life_prime_info.pickle"
                    )
                    shutil.copyfile(stat_file, live_prime_info_file)

        # Convert space_group_info object to str, to make compatible with JSON
        if prime_info and "space_group_info" in prime_info:
            for i in prime_info["space_group_info"]:
                sg = str(i).split("(")[0].rstrip(" ")
                idx = prime_info["space_group_info"].index(i)
                prime_info["space_group_info"][idx] = sg

        return prime_info

    def abort(self):
        # TODO: put in an LSF kill command
        if hasattr(self, "job"):
            try:
                self.job.kill_thread()
            except Exception as e:
                print("PRIME THREAD ERROR: Cannot terminate thread! {}".format(e))

    def run(self):
        # Generate PRIME input
        cmd_args = self.prepare_PRIME_input()

        # Run PRIME
        if cmd_args:
            # remove previous run to avoid conflict
            prime_dir = os.path.join(self.info.int_base, "prime/000")
            if os.path.isdir(prime_dir):
                shutil.rmtree(prime_dir)

            # Launch PRIME
            prime_file = os.path.join(self.info.int_base, "live_prime.phil")
            cmd = "prime.run {} {}".format(prime_file, cmd_args)

            try:
                # easy_run.fully_buffered(cmd, join_stdout_stderr=True).show_stdout()
                self.job = CustomRun(command=cmd, join_stdout_stderr=True)
                self.job.run()
                prime_info = self.get_prime_stats()
            except Exception as e:
                print("LIVE PRIME ERROR: ", e)
                prime_info = None

        else:
            prime_info = None

        # Send signal to UI
        evt = PRIMEAllDone(tp_EVT_PRIMEDONE, -1, info=prime_info)
        wx.PostEvent(self.parent, evt)


tp_EVT_SLICEDONE = wx.NewEventType()
EVT_SLICEDONE = wx.PyEventBinder(tp_EVT_SLICEDONE)
tp_EVT_CLUSTERDONE = wx.NewEventType()
EVT_CLUSTERDONE = wx.PyEventBinder(tp_EVT_CLUSTERDONE, 1)


class ClusteringDone(wx.PyCommandEvent):
    """Send event when finished all cycles."""

    def __init__(self, etype, eid, info=None):
        wx.PyCommandEvent.__init__(self, etype, eid)
        self.info = info

    def GetValue(self):
        return self.info


class ClusterWorkThread:
    def __init__(self, parent):
        self.parent = parent
        self.abort = False

    def run(self, iterable):

        # with Capturing() as junk_output:
        errors = []
        try:
            ucs = Cluster.from_iterable(iterable=iterable)
            clusters, _ = ucs.ab_cluster(
                5000, log=False, write_file_lists=False, schnell=True, doplot=False
            )
        except Exception as e:
            print("IOTA ERROR (CLUSTERING): ", e)
            clusters = []
            errors.append(str(e))

        info = []
        if clusters:
            for cluster in clusters:
                uc_init = unit_cell(cluster.medians)
                symmetry = crystal.symmetry(unit_cell=uc_init, space_group_symbol="P1")
                groups = lattice_symmetry.metric_subgroups(
                    input_symmetry=symmetry, max_delta=3
                )
                top_group = groups.result_groups[0]
                best_sg = str(groups.lattice_group_info()).split("(")[0]
                best_uc = top_group["best_subsym"].unit_cell().parameters()
                uc_no_stdev = (
                    "{:<6.2f} {:<6.2f} {:<6.2f} "
                    "{:<6.2f} {:<6.2f} {:<6.2f} "
                    "".format(
                        best_uc[0],
                        best_uc[1],
                        best_uc[2],
                        best_uc[3],
                        best_uc[4],
                        best_uc[5],
                    )
                )
                cluster_info = {
                    "number": len(cluster.members),
                    "pg": str(best_sg),
                    "uc": uc_no_stdev,
                }
                info.append(cluster_info)

        return info, errors


class ClusterThread(Thread):
    """Basic spotfinder (with defaults) that could be used to rapidly analyze
    images as they are collected."""

    def __init__(self, parent, iterable):
        Thread.__init__(self, name="live_cluster")
        self.parent = parent
        self.iterable = iterable
        self.clustering = ClusterWorkThread(self)

    def abort(self):
        self.clustering.abort = True

    def run(self):
        clusters, errors = self.clustering.run(iterable=self.iterable)

        if clusters:
            clusters = sorted(clusters, key=lambda i: i["number"], reverse=True)
        evt = SpotFinderOneDone(tp_EVT_CLUSTERDONE, -1, info=[clusters, errors])
        wx.PostEvent(self.parent, evt)
