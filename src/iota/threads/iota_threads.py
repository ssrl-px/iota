from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 02/08/2021
Last Changed: 02/08/2021
Description : IOTA Threads and PostEvents
"""

import os
import signal
import subprocess
import sys
from threading import Thread

import wx
from libtbx import easy_pickle as ep, easy_run

from iota.utils.input_finder import InputFinder
from iota.analysis.iota_analysis import Analyzer

ginp = InputFinder()
tp_EVT_ALLDONE = wx.NewEventType()
EVT_ALLDONE = wx.PyEventBinder(tp_EVT_ALLDONE, 1)
tp_EVT_IMGDONE = wx.NewEventType()
EVT_IMGDONE = wx.PyEventBinder(tp_EVT_IMGDONE, 1)
tp_EVT_OBJDONE = wx.NewEventType()
EVT_OBJDONE = wx.PyEventBinder(tp_EVT_OBJDONE, 1)


class ImageFinderAllDone(wx.PyCommandEvent):
    """Send event when finished all cycles."""

    def __init__(self, etype, eid, input_list=None):
        wx.PyCommandEvent.__init__(self, etype, eid)
        self.input_list = input_list

    def GetValue(self):
        return self.input_list


class ObjectFinderAllDone(wx.PyCommandEvent):
    """Send event when finished all cycles."""

    def __init__(self, etype, eid, obj_list=None):
        wx.PyCommandEvent.__init__(self, etype, eid)
        self.obj_list = obj_list

    def GetValue(self):
        return self.obj_list


class AllDone(wx.PyCommandEvent):
    """Send event when finished all cycles."""

    def __init__(self, etype, eid, img_objects=None):
        wx.PyCommandEvent.__init__(self, etype, eid)
        self.image_objects = img_objects

    def GetValue(self):
        return self.image_objects


class JobSubmitThread(Thread):
    """Thread for easy_run submissions so that they don't block GUI."""

    def __init__(self, parent, params, out_type="gui_silent"):
        Thread.__init__(self)
        self.parent = parent
        self.params = params
        self.job_id = None

        # Enforce silent run if on mp queue (unless need debug statements)
        if self.params.mp.method != "multiprocessing" and out_type != "gui_debug":
            self.out_type = "gui_silent"
        else:
            self.out_type = out_type

    def submit(self):
        run_path = self.parent.info.int_base
        iota_cmd = "iota.run --run_path {} -o {}".format(run_path, self.out_type)
        if self.params.mp.submit_command:
            command = self.params.mp.submit_command.replace("<iota_command>", iota_cmd)
        else:
            command = iota_cmd

        if command is not None:
            print(command)
            self.job = CustomRun(command=str(command), join_stdout_stderr=True)
            self.job.run()
            self.job.show_stdout()
            if self.job_id is not None:
                print("JOB NAME = ", self.job_id)
            return
        else:
            print("IOTA ERROR: COMMAND NOT ISSUED!")
            return

    def abort(self):
        if self.params.mp.kill_command:
            CustomRun(command=self.params.mp.kill_command)
        else:
            try:
                self.job.kill_thread()
            except Exception as e:
                print("IOTA JOB ERROR: Cannot kill job thread! {}".format(e))

    def run(self):
        return self.submit()


class ObjectReader:
    def __init__(self):
        pass

    def update_info(self, info):
        # # Determine chunk of image objects to check for existing
        # if self.n_proc:
        #   chunk = self.n_proc
        # else:
        #   chunk = 25
        # if len(self.info.unread_files) < chunk:
        #   chunk = len(self.info.unread_files)
        #
        # # Create filelist of written image objects
        # filelist = []
        # for fp in self.info.unread_files[:chunk]:
        #   if os.path.isfile(fp):
        #     filelist.append(fp)
        #     self.info.unread_files.pop(fp)
        #
        # # Perform stat extraction
        # if filelist:
        #   from iota.analysis.iota_analysis import Analyzer
        #   analyzer = Analyzer(info=self.info)
        #   stats_OK = analyzer.get_results(filelist=filelist)
        #   if stats_OK:
        #     self.info = analyzer.info
        #     self.obs = analyzer.obs

        finished_objects = info.get_finished_objects_from_file()
        if finished_objects:
            analyzer = Analyzer(info=info, gui_mode=True)
            stats_OK = analyzer.run_get_results(finished_objects=finished_objects)
            if stats_OK:
                return analyzer.info
        return None

    def run(self, info):
        return self.update_info(info)


class ObjectReaderThread(Thread):
    """Thread for reading processed objects and making all calculations for
    plotting in main GUI thread."""

    def __init__(self, parent, info=None):
        Thread.__init__(self, name="object_reader")
        self.parent = parent
        self.info = info
        self.obj_worker = ObjectReader()

    def run(self):
        info = self.obj_worker.run(self.info)
        if info:
            info.export_json()
            evt = ObjectFinderAllDone(tp_EVT_OBJDONE, -1, obj_list=info)
            wx.PostEvent(self.parent, evt)


class ImageFinderThread(Thread):
    """Worker thread generated to poll filesystem on timer.

    Will check to see if any new images have been found. Put on a thread
    to run in background
    """

    def __init__(
        self,
        parent,
        input,
        input_list,
        min_back=None,
        last_file=None,
        back_to_thread=False,
    ):
        Thread.__init__(self)
        self.parent = parent
        self.input = input
        self.min_back = min_back
        self.last_file = last_file
        self.back_to_thread = back_to_thread

        # Generate comparable input list
        self.input_list = []
        for item in input_list:
            if isinstance(item, list) or isinstance(item, tuple):
                self.input_list.append((item[1], item[2]))
            else:
                self.input_list.append(item)

    def run(self):
        # Poll filesystem and determine which files are new (if any)

        ext_file_list, _ = ginp.make_input_list(
            self.input,
            filter_results=True,
            filter_type="image",
            min_back=self.min_back,
            last=self.last_file,
            expand_multiple=True,
        )

        new_input_list = list(set(ext_file_list) - set(self.input_list))
        new_input_list = sorted(new_input_list)

        if self.back_to_thread:
            wx.CallAfter(self.parent.onImageFinderDone, new_input_list)
        else:
            evt = ImageFinderAllDone(tp_EVT_IMGDONE, -1, input_list=new_input_list)
            wx.PostEvent(self.parent, evt)


class ObjectFinderThread(Thread):
    """Worker thread that polls filesystem on timer for image objects.

    Will collect and extract info on images processed so far
    """

    def __init__(self, parent, object_folder, last_object=None, new_fin_base=None):
        Thread.__init__(self)
        self.parent = parent
        self.object_folder = object_folder
        self.new_fin_base = new_fin_base
        self.last_object = last_object

    def run(self):
        if self.last_object is not None:
            last = self.last_object.obj_file
        else:
            last = None
        object_files, _ = ginp.get_input_from_folder(
            self.object_folder, ext_only="int", last=last
        )
        new_objects = [self.read_object_file(i) for i in object_files]
        new_finished_objects = [i for i in new_objects if i is not None]

        evt = ObjectFinderAllDone(tp_EVT_OBJDONE, -1, obj_list=new_finished_objects)
        wx.PostEvent(self.parent, evt)

    def read_object_file(self, filepath):
        try:
            object = ep.load(filepath)
            return object
        except EOFError as e:
            print("OBJECT_IMPORT_ERROR: ", e)
            return None


class ImageViewerThread(Thread):
    """Worker thread that will move the image viewer launch away from the GUI
    and hopefully will prevent the image selection dialog freezing on MacOS."""

    def __init__(
        self, parent, file_string=None, viewer="dials.image_viewer", img_type=None
    ):
        Thread.__init__(self)
        self.parent = parent
        self.file_string = file_string
        self.viewer = viewer
        self.img_type = img_type
        self.options = (
            "show_ctr_mass=False "
            "show_max_pix=False "
            "show_all_pix=False "
            "show_predictions=False "
            "show_basis_vectors=False"
        )

    def run(self):
        command = "{} {} {}".format(self.viewer, self.file_string, self.options)
        easy_run.fully_buffered(command)


class CustomRun(easy_run.fully_buffered_base):
    """A subclass from easy_run with a "kill switch" for easy process
    termination from UI; took out timeout, since that won't be used, and
    doesn't work on all systems, anyway.

    Tested on Mac OS X 10.13.6 and CentOS 6 so far.
    """

    def __init__(
        self,
        command,
        timeout=None,
        stdin_lines=None,
        join_stdout_stderr=False,
        stdout_splitlines=True,
        bufsize=-1,
    ):
        self.command = command
        self.join_stdout_stderr = join_stdout_stderr
        self.timeout = timeout
        self.stdin_lines = stdin_lines
        self.stdout_splitlines = stdout_splitlines
        self.bufsize = bufsize
        self.thread = None

    def target(self, process, lines, result):
        o, e = process.communicate(input=lines)
        result[0] = o
        result[1] = e

    def kill_thread(self):
        os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)

    def run(self):
        if not isinstance(self.command, str):
            self.command = subprocess.list2cmdline(self.command)
        if sys.platform == "darwin":  # bypass SIP on OS X 10.11
            self.command = (
                "DYLD_LIBRARY_PATH=%s exec " % os.environ.get("DYLD_LIBRARY_PATH", "")
            ) + self.command
        if self.stdin_lines is not None:
            if not isinstance(self.stdin_lines, str):
                self.stdin_lines = os.linesep.join(self.stdin_lines)
                if len(self.stdin_lines) != 0:
                    self.stdin_lines += os.linesep
        if self.join_stdout_stderr:
            stderr = subprocess.STDOUT
        else:
            stderr = subprocess.PIPE

        self.p = subprocess.Popen(
            args=self.command,
            shell=True,
            bufsize=self.bufsize,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            universal_newlines=True,
            close_fds=(sys.platform != "win32"),
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )
        o, e = self.p.communicate(input=self.stdin_lines)
        if self.stdout_splitlines:
            self.stdout_buffer = None
            self.stdout_lines = o.splitlines()
        else:
            self.stdout_buffer = o
            self.stdout_lines = None
        if self.join_stdout_stderr:
            self.stderr_lines = []
        else:
            self.stderr_lines = e.splitlines()
        self.return_code = self.p.returncode
