from __future__ import division

"""
Author      : Lyubimov, A.Y.
Created     : 01/17/2017
Last Changed: 01/17/2017
Description : IOTA GUI Windows / frames
"""

import os
import wx
from wxtbx import bitmaps

import math
import numpy as np
import time
import warnings
import multiprocessing

import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

from libtbx import easy_pickle as ep
from libtbx.utils import to_unicode
from cctbx import miller

assert miller
from libtbx import easy_run

from iota.components.iota_utils import GenerateInput, get_file_list
from iota.components.iota_analysis import Analyzer, Plotter
from iota.components.iota_misc import WxFlags, noneset
import iota.components.iota_controls as ct
import iota.components.iota_threads as thr
import iota.components.iota_dialogs as dlg

f = WxFlags()

# Platform-specific stuff
# TODO: Will need to test this on Windows at some point
if wx.Platform == "__WXGTK__":
    norm_font_size = 10
    button_font_size = 12
    LABEL_SIZE = 14
    CAPTION_SIZE = 12
elif wx.Platform == "__WXMAC__":
    norm_font_size = 12
    button_font_size = 14
    LABEL_SIZE = 14
    CAPTION_SIZE = 12
elif wx.Platform == "__WXMSW__":
    norm_font_size = 9
    button_font_size = 11
    LABEL_SIZE = 11
    CAPTION_SIZE = 9

# ------------------------------ Input Window -------------------------------- #


class BasePanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY, size=(800, 500))

        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)


class InputWindow(BasePanel):
    """ Input window - data input, description of project """

    def __init__(self, parent, phil):
        BasePanel.__init__(self, parent=parent)

        # Generate default parameters
        self.input_phil = phil
        self.target_phil = None
        self.gparams = self.input_phil.extract()

        if str(self.gparams.advanced.integrate_with).lower() == "cctbx":
            target = self.gparams.cctbx.target
        elif str(self.gparams.advanced.integrate_with).lower() == "dials":
            target = self.gparams.dials.target

        if str(target).lower != "none":
            try:
                with open(target, "r") as pf:
                    self.target_phil = pf.read()
            except Exception:
                self.target_phil = None

        self.int_box = ct.ChoiceCtrl(
            self,
            label="Integrate with:",
            label_size=(150, -1),
            choices=["cctbx", "DIALS"],
            ctrl_size=(200, -1),
        )
        self.int_box.ctr.SetSelection(0)

        self.project_folder = ct.InputCtrl(
            self,
            label="Project Folder: ",
            label_size=(150, -1),
            label_style="bold",
            value=os.path.abspath(os.curdir),
            buttons=True,
        )

        self.project_title = ct.InputCtrl(
            self, label="Description", label_size=(150, -1), label_style="normal"
        )

        # List control to add / manage input items
        self.input = FileListCtrl(self)

        # Put everything into main sizer
        self.main_sizer.Add(self.int_box, flag=f.expand, border=10)
        self.main_sizer.Add(self.project_title, flag=f.expand, border=10)
        self.main_sizer.Add(self.project_folder, flag=f.expand, border=10)
        self.main_sizer.Add(self.input, 1, flag=wx.EXPAND | wx.ALL, border=10)
        self.main_sizer.Add(wx.StaticLine(self), flag=wx.EXPAND)

        # Options
        opt_box = wx.FlexGridSizer(1, 5, 0, 15)
        total_procs = multiprocessing.cpu_count()
        self.opt_spc_nprocs = ct.SpinCtrl(
            self,
            label="No. Processors: ",
            label_size=(120, -1),
            ctrl_max=total_procs,
            ctrl_min=1,
            ctrl_value=str(int(total_procs / 2)),
        )
        self.opt_btn_import = wx.Button(self, label="Import options...")
        self.opt_btn_process = wx.Button(self, label="Processing options...")
        self.opt_btn_analysis = wx.Button(self, label="Analysis options...")
        opt_box.AddMany(
            [
                (self.opt_spc_nprocs),
                (0, 0),
                (self.opt_btn_import),
                (self.opt_btn_process),
                (self.opt_btn_analysis),
            ]
        )
        opt_box.AddGrowableCol(1)
        self.main_sizer.Add(opt_box, flag=f.expand, border=10)

        # Button bindings
        self.project_folder.btn_browse.Bind(wx.EVT_BUTTON, self.onOutputBrowse)
        self.project_folder.btn_mag.Bind(wx.EVT_BUTTON, self.onMagButton)
        self.Bind(wx.EVT_BUTTON, self.onImportOptions, self.opt_btn_import)
        self.Bind(wx.EVT_BUTTON, self.onProcessOptions, self.opt_btn_process)
        self.Bind(wx.EVT_BUTTON, self.onAnalysisOptions, self.opt_btn_analysis)

    def onMagButton(self, e):
        dirview = dlg.DirView(self, title="Current Folder")
        if dirview.ShowModal() == wx.ID_OK:
            dirview.Destroy()

    def onImportOptions(self, e):
        e.Skip()

    def onProcessOptions(self, e):
        e.Skip()

    def onAnalysisOptions(self, e):
        e.Skip()

    def onInfo(self, e):
        """On clicking the info button."""
        info_txt = """Input diffraction images here. IOTA accepts either raw images (mccd, cbf, img, etc.) or image pickles. Input can be either a folder with images, or a text file with a list of images."""
        info = wx.MessageDialog(None, info_txt, "Info", wx.OK)
        info.ShowModal()

    def onOutputBrowse(self, e):
        """On clicking the Browse button: show the DirDialog and populate
        'Output' box w/ selection."""
        dlg = wx.DirDialog(
            self, "Choose the output directory:", style=wx.DD_DEFAULT_STYLE
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.project_folder.ctr.SetValue(dlg.GetPath())
        dlg.Destroy()
        e.Skip()


class FileListCtrl(ct.CustomListCtrl):
    """File list window for the input tab."""

    def __init__(self, parent, size=(-1, 300)):
        ct.CustomListCtrl.__init__(self, parent=parent, size=size)

        self.parent = parent
        self.main_window = parent.GetParent()

        # Initialize dictionaries for imported data types
        self.all_data_images = {}
        self.all_img_objects = {}
        self.all_proc_pickles = {}

        # Generate columns
        self.ctr.InsertColumn(0, "Path")
        self.ctr.InsertColumn(1, "Input Type")
        self.ctr.InsertColumn(2, "Action")
        self.ctr.setResizeColumn(1)

        # Add file / folder buttons
        self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_add_file = wx.Button(self, label="Add File...")
        self.btn_add_dir = wx.Button(self, label="Add Folder...")
        self.button_sizer.Add(self.btn_add_file)
        self.button_sizer.Add(self.btn_add_dir, flag=wx.LEFT, border=10)

        self.sizer.Add(self.button_sizer, flag=wx.TOP | wx.BOTTOM, border=10)

        # Event bindings
        self.Bind(wx.EVT_BUTTON, self.onAddFile, self.btn_add_file)
        self.Bind(wx.EVT_BUTTON, self.onAddFolder, self.btn_add_dir)

    def view_all_images(self):
        if self.ctr.GetItemCount() > 0:
            file_list = []
            for i in range(self.ctr.GetItemCount()):
                type_ctrl = self.ctr.GetItemWindow(i, col=1).type
                type_choice = type_ctrl.GetString(type_ctrl.GetSelection())
                if type_choice in ("raw images folder", "image pickles folder"):
                    for root, dirs, files in os.walk(self.ctr.GetItemText(i)):
                        for filename in files:
                            file_list.append(os.path.join(root, filename))
                if type_choice in ("image pickle", "raw image"):
                    file_list.append(self.ctr.GetItemText(i))
                if type_choice == "image list":
                    with open(self.ctr.GetItemText(i), "r") as lf:
                        file_list.append(lf.readlines())

            try:
                file_string = " ".join(file_list)
                easy_run.fully_buffered("cctbx.image_viewer {}".format(file_string))
            except Exception, e:
                print e

        else:
            wx.MessageBox("No data found", "Error", wx.OK | wx.ICON_ERROR)

    def onAddFile(self, e):
        file_dlg = wx.FileDialog(
            self,
            message="Load File",
            defaultDir=os.curdir,
            defaultFile="*",
            wildcard="*",
            style=wx.OPEN | wx.FD_FILE_MUST_EXIST,
        )
        if file_dlg.ShowModal() == wx.ID_OK:
            self.add_item(file_dlg.GetPaths()[0])
        file_dlg.Destroy()
        e.Skip()

    def onAddFolder(self, e):
        dlg = wx.DirDialog(self, "Load Folder:", style=wx.DD_DEFAULT_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            self.add_item(dlg.GetPath())
        dlg.Destroy()
        e.Skip()

    def set_type_choices(self, path):
        # Determine what type of input this is and present user with choices
        # (this so far works for images ONLY)
        ginp = GenerateInput()
        type_choices = ["[  SELECT INPUT TYPE  ]"]
        preferred_selection = 0
        if os.path.isdir(path):
            type_choices.extend(["raw images folder", "image pickles folder"])
            dir_type = ginp.get_folder_type(path)
            if dir_type in type_choices:
                preferred_selection = type_choices.index(dir_type)
        elif os.path.isfile(path):
            file_type = ginp.get_file_type(path)
            if file_type in ("image pickle", "raw image"):
                type_choices.extend(["raw image", "image pickle"])
                if file_type in type_choices:
                    preferred_selection = type_choices.index(file_type)
            elif file_type == "image list":
                type_choices.extend(["image list"])
                preferred_selection = type_choices.index("image list")
            elif file_type in (
                "IOTA settings",
                "PRIME settings",
                "LABELIT target",
                "DIALS target",
            ):
                type_choices.extend(
                    [
                        "IOTA settings",
                        "PRIME settings",
                        "LABELIT target",
                        "DIALS target",
                    ]
                )
                preferred_selection = type_choices.index(file_type)

        return type_choices, preferred_selection

    def add_item(self, path):
        # Generate item
        inp_choices, inp_sel = self.set_type_choices(path)
        type_choice = ct.DataTypeChoice(self.ctr, choices=inp_choices)
        item = ct.InputListItem(
            path=path, type=type_choice, buttons=ct.MiniButtonBoxInput(self.ctr)
        )

        self.Bind(wx.EVT_CHOICE, self.onTypeChoice, item.type.type)
        self.Bind(wx.EVT_BUTTON, self.onMagButton, item.buttons.btn_mag)
        self.Bind(wx.EVT_BUTTON, self.onDelButton, item.buttons.btn_delete)
        self.Bind(wx.EVT_BUTTON, self.onInfoButton, item.buttons.btn_info)

        # Insert list item
        idx = self.ctr.InsertStringItem(self.ctr.GetItemCount() + 1, item.path)
        self.ctr.SetItemWindow(idx, 1, item.type, expand=True)
        self.ctr.SetItemWindow(idx, 2, item.buttons, expand=True)

        # Set drop-down selection, check it for data and open other tabs
        item.type.type.SetSelection(inp_sel)
        if item.type.type.GetString(inp_sel) in [
            "raw images folder",
            "image pickles folder",
            "image pickle",
            "raw image",
            "image list",
        ]:
            self.main_window.toolbar.EnableTool(
                self.main_window.tb_btn_run.GetId(), True
            )
            if os.path.isdir(item.path):
                ignore_ext = (
                    "txt",
                    "log",
                    "lst",
                    "seq",
                    "phil",
                    "param",
                    "inp",
                    "int",
                    "tmp",
                    "png",
                    "jpg",
                    "jpeg",
                )
                self.all_data_images[item.path] = get_file_list(
                    item.path, ignore_ext=ignore_ext
                )
            elif os.path.isfile(item.path):
                if item.type.type.GetString(inp_sel) == "image list":
                    with open(item.path, "r") as f:
                        items = f.readlines()
                    self.all_data_images[item.path] = items
                else:
                    self.all_data_images[item.path] = [item.path]
            if item.type.type.GetString(inp_sel) in ("image pickle", "raw image"):
                view_bmp = bitmaps.fetch_custom_icon_bitmap("image_viewer16")
                item.buttons.btn_mag.SetBitmapLabel(view_bmp)
        else:
            warn_bmp = bitmaps.fetch_icon_bitmap("actions", "status_unknown", size=16)
            item.buttons.btn_info.SetBitmapLabel(warn_bmp)
            item.warning = True

        # Record index in all relevant places
        item.id = idx
        item.buttons.index = idx
        item.type.index = idx
        item.type_selection = inp_sel

        # Resize columns to fit content
        col1_width = (
            max(
                [
                    self.ctr.GetItemWindow(s, col=1).type.GetSize()[0]
                    for s in range(self.ctr.GetItemCount())
                ]
            )
            + 5
        )
        col2_width = item.buttons.GetSize()[0] + 15
        col0_width = self.ctr.GetSize()[0] - col1_width - col2_width
        self.ctr.SetColumnWidth(0, col0_width)
        self.ctr.SetColumnWidth(1, col1_width)
        self.ctr.SetColumnWidth(2, col2_width)

        # Make sure all the choice lists are the same size
        if item.type.type.GetSize()[0] < col1_width - 5:
            item.type.type.SetSize((col1_width - 5, -1))

        # Attach data object to item
        self.ctr.SetItemData(item.id, item)

    def onTypeChoice(self, e):
        type = e.GetEventObject().GetParent()
        item_data = self.ctr.GetItemData(type.index)
        item_data.type.type.SetSelection(type.type.GetSelection())
        item_data.type_selection = type.type.GetSelection()

        # Evaluate whether data folders / files are present
        data_items = 0
        for idx in range(self.ctr.GetItemCount()):
            if self.ctr.GetItemData(idx).type_selection != 0:
                data_items += 1
        if data_items > 0:
            self.main_window.toolbar.EnableTool(
                self.main_window.tb_btn_run.GetId(), True
            )
        else:
            self.main_window.toolbar.EnableTool(
                self.main_window.tb_btn_run.GetId(), False
            )
        e.Skip()

    def onMagButton(self, e):
        idx = e.GetEventObject().GetParent().index
        item_obj = self.ctr.GetItemData(idx)
        path = item_obj.path
        ginp = GenerateInput()

        if os.path.isfile(path):
            if ginp.get_file_type(path) in ("raw image", "image pickle"):
                self.view_images(path)
            elif ginp.is_text(path):
                with open(path, "r") as f:
                    msg = f.read()
                textview = dlg.TextFileView(self, title=path, contents=msg)
                textview.ShowModal()
            elif ginp.get_file_type(path) == "binary":
                wx.MessageBox(
                    "Unknown binary file", "Warning", wx.OK | wx.ICON_EXCLAMATION
                )
        elif os.path.isdir(path):
            file_list = ""
            for root, dirs, files in os.walk(path):
                for filename in files:
                    found_file = os.path.join(root, filename)
                    file_list += "{}\n".format(found_file)
            filelistview = dlg.TextFileView(self, title=path, contents=file_list)
            filelistview.ShowModal()

    def view_images(self, img_list):
        """Launches image viewer (depending on backend) with either one image,
        multiple images, or many images (100 limit)"""
        file_string = " ".join(img_list)
        backend = str(self.parent.gparams.advanced.integrate_with).lower()
        easy_run.fully_buffered("{}.image_viewer {}".format(backend, file_string))

    def onDelButton(self, e):
        item = e.GetEventObject().GetParent()
        self.delete_button(item.index)

    def delete_all(self):
        for idx in range(self.ctr.GetItemCount()):
            self.delete_button(index=0)

    def delete_button(self, index):
        self.ctr.DeleteItem(index)

        # Refresh widget and list item indices
        for i in range(self.ctr.GetItemCount()):
            item_data = self.ctr.GetItemData(i)
            item_data.id = i
            item_data.buttons.index = i
            item_data.type.index = i
            type_choice = self.ctr.GetItemWindow(i, col=1)
            type_selection = item_data.type.type.GetSelection()
            type_choice.type.SetSelection(type_selection)
            self.ctr.SetItemData(i, item_data)

    def onInfoButton(self, e):
        """Info / alert / error button (will change depending on
        circumstance)"""
        idx = e.GetEventObject().GetParent().index
        item_obj = self.ctr.GetItemData(idx)
        item_type = item_obj.type.type.GetString(item_obj.type_selection)

        if item_obj.warning:
            wx.MessageBox(
                item_obj.info["WARNING"], "Warning", wx.OK | wx.ICON_EXCLAMATION
            )
        else:
            wx.MessageBox(item_obj.info[item_type], "Info", wx.OK | wx.ICON_INFORMATION)


# ----------------------------  Processing Window ---------------------------  #


class LogTab(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.log_sizer = wx.BoxSizer(wx.VERTICAL)
        self.log_window = wx.TextCtrl(
            self, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP
        )
        self.log_window.SetFont(wx.Font(9, wx.TELETYPE, wx.NORMAL, wx.NORMAL, False))
        self.log_sizer.Add(
            self.log_window, proportion=1, flag=wx.EXPAND | wx.ALL, border=10
        )
        self.SetSizer(self.log_sizer)


class ProcessingTab(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.proc_sizer = wx.BoxSizer(wx.VERTICAL)
        self.proc_figure = Figure()

        # Create transparent background
        self.proc_figure.patch.set_alpha(0)

        # Set regular font
        plt.rc("font", family="sans-serif", size=10)
        plt.rc("mathtext", default="regular")

        gsp = gridspec.GridSpec(4, 4)
        self.int_axes = self.proc_figure.add_subplot(gsp[2:, 2:])
        self.int_axes.axis("off")
        self.bxy_axes = self.proc_figure.add_subplot(gsp[2:, :2])

        gsub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gsp[:2, :], hspace=0)

        self.nsref_axes = self.proc_figure.add_subplot(gsub[1])
        self.res_axes = self.proc_figure.add_subplot(gsub[0])
        self.nsref_axes.set_xlabel("Frame")
        self.nsref_axes.set_ylabel("Strong Spots")
        self.nsref_axes.yaxis.get_major_ticks()[0].label1.set_visible(False)
        self.nsref_axes.yaxis.get_major_ticks()[-1].label1.set_visible(False)
        self.res_axes.set_ylabel("Resolution")
        self.res_axes.yaxis.get_major_ticks()[0].label1.set_visible(False)
        self.res_axes.yaxis.get_major_ticks()[-1].label1.set_visible(False)
        plt.setp(self.res_axes.get_xticklabels(), visible=False)
        self.proc_figure.set_tight_layout(True)
        self.canvas = FigureCanvas(self, -1, self.proc_figure)
        self.proc_sizer.Add(
            self.canvas, proportion=1, flag=wx.EXPAND | wx.ALL, border=10
        )
        self.SetSizer(self.proc_sizer)


class SummaryTab(wx.Panel):
    def __init__(
        self, parent, gparams=None, final_objects=None, out_dir=None, plot=None
    ):
        wx.Panel.__init__(self, parent)

        self.final_objects = final_objects
        self.gparams = gparams
        self.out_dir = out_dir
        self.plot = plot

        summary_sizer = wx.BoxSizer(wx.VERTICAL)

        sfont = wx.Font(norm_font_size, wx.DEFAULT, wx.NORMAL, wx.NORMAL)
        bfont = wx.Font(norm_font_size, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        self.SetFont(bfont)

        # Run information
        run_box = wx.StaticBox(self, label="Run Information")
        run_box.SetFont(sfont)
        run_box_sizer = wx.StaticBoxSizer(run_box, wx.VERTICAL)
        run_box_grid = wx.FlexGridSizer(3, 2, 5, 20)
        self.title_txt = wx.StaticText(self, label="")
        self.title_txt.SetFont(sfont)
        self.folder_txt = wx.StaticText(self, label="")
        self.folder_txt.SetFont(sfont)

        run_box_grid.AddMany(
            [
                (wx.StaticText(self, label="Title")),
                (self.title_txt, 1, wx.EXPAND),
                (wx.StaticText(self, label="Directory")),
                (self.folder_txt, 1, wx.EXPAND),
            ]
        )

        run_box_grid.AddGrowableCol(1, 1)
        run_box_sizer.Add(run_box_grid, flag=wx.EXPAND | wx.ALL, border=10)

        summary_sizer.Add(run_box_sizer, flag=wx.EXPAND | wx.ALL, border=10)

        # Integration summary
        if self.gparams.advanced.integrate_with == "cctbx":
            int_box = wx.StaticBox(self, label="Analysis of Integration")
            int_box.SetFont(sfont)
            int_box_sizer = wx.StaticBoxSizer(int_box, wx.HORIZONTAL)
            int_box_grid = wx.FlexGridSizer(4, 5, 5, 20)
            int_btn_sizer = wx.BoxSizer(wx.VERTICAL)

            # Grid search summary
            self.sih_min = wx.StaticText(self, label="4.0")
            self.sih_min.SetFont(sfont)
            self.sih_max = wx.StaticText(self, label="8.0")
            self.sih_max.SetFont(sfont)
            self.sih_avg = wx.StaticText(self, label="6.0")
            self.sih_avg.SetFont(sfont)
            self.sih_std = wx.StaticText(self, label="0.05")
            self.sih_std.SetFont(sfont)
            self.sph_min = wx.StaticText(self, label="5.0")
            self.sph_min.SetFont(sfont)
            self.sph_max = wx.StaticText(self, label="12.0")
            self.sph_max.SetFont(sfont)
            self.sph_avg = wx.StaticText(self, label="8.5")
            self.sph_avg.SetFont(sfont)
            self.sph_std = wx.StaticText(self, label="0.15")
            self.sph_std.SetFont(sfont)
            self.spa_min = wx.StaticText(self, label="10.0")
            self.spa_min.SetFont(sfont)
            self.spa_max = wx.StaticText(self, label="20.0")
            self.spa_max.SetFont(sfont)
            self.spa_avg = wx.StaticText(self, label="15.0")
            self.spa_avg.SetFont(sfont)
            self.spa_std = wx.StaticText(self, label="0.01")
            self.spa_std.SetFont(sfont)

            int_box_grid.AddMany(
                [
                    (wx.StaticText(self, label="")),
                    (wx.StaticText(self, label="min")),
                    (wx.StaticText(self, label="max")),
                    (wx.StaticText(self, label="avg")),
                    (wx.StaticText(self, label="std")),
                    (wx.StaticText(self, label="minimum signal height")),
                    (self.sih_min),
                    (self.sih_max),
                    (self.sih_avg),
                    (self.sih_std),
                    (wx.StaticText(self, label="minimum spot height")),
                    (self.sph_min),
                    (self.sph_max),
                    (self.sph_avg),
                    (self.sph_std),
                    (wx.StaticText(self, label="minimum spot area")),
                    (self.spa_min),
                    (self.spa_max),
                    (self.spa_avg),
                    (self.spa_std),
                ]
            )

            # Button & binding for heatmap display
            heatmap_bmp = bitmaps.fetch_custom_icon_bitmap("heatmap24")
            self.int_heatmap = ct.GradButton(
                self, bmp=heatmap_bmp, label="  Spotfinding Heatmap", size=(250, -1)
            )
            int_btn_sizer.Add(self.int_heatmap)
            self.Bind(wx.EVT_BUTTON, self.onPlotHeatmap, self.int_heatmap)

            # Insert into sizers
            int_box_sizer.Add(int_box_grid, flag=wx.ALL, border=10)
            int_box_sizer.AddStretchSpacer()
            int_box_sizer.Add(int_btn_sizer, flag=wx.ALL, border=10)
            summary_sizer.Add(int_box_sizer, flag=wx.EXPAND | wx.ALL, border=10)

        # Dataset Info
        dat_box = wx.StaticBox(self, label="Dataset Information")
        dat_box.SetFont(sfont)
        dat_box_sizer = wx.StaticBoxSizer(dat_box, wx.HORIZONTAL)
        dat_box_grid = wx.FlexGridSizer(4, 2, 5, 20)
        dat_btn_sizer = wx.BoxSizer(wx.VERTICAL)

        self.pg_txt = wx.StaticText(self, label="P4")
        self.pg_txt.SetFont(sfont)
        self.uc_txt = wx.StaticText(self, label="79 79 38 90 90 90")
        self.uc_txt.SetFont(sfont)
        res = to_unicode(u"50.0 - 1.53 {}".format(u"\u212B"))
        self.rs_txt = wx.StaticText(self, label=res)
        self.rs_txt.SetFont(sfont)
        self.xy_txt = wx.StaticText(self, label="X = 224.90 mm, Y = 225.08 mm")
        self.xy_txt.SetFont(sfont)

        dat_box_grid.AddMany(
            [
                (wx.StaticText(self, label="Bravais lattice: ")),
                (self.pg_txt),
                (wx.StaticText(self, label="Unit cell: ")),
                (self.uc_txt),
                (wx.StaticText(self, label="Resolution: ")),
                (self.rs_txt),
                (wx.StaticText(self, label="Beam XY: ")),
                (self.xy_txt),
            ]
        )

        # Buttons for res. histogram and beam xy plot
        hist_bmp = bitmaps.fetch_icon_bitmap(
            "mimetypes", "spreadsheet", size=32, scale=(24, 24)
        )
        self.dat_reshist = ct.GradButton(
            self, bmp=hist_bmp, label="  Resolution Histogram", size=(250, -1)
        )
        beamXY_bmp = bitmaps.fetch_custom_icon_bitmap("scatter_plot_24")
        self.dat_beamxy = ct.GradButton(
            self, bmp=beamXY_bmp, label="  Beam XY Plot", size=(250, -1)
        )
        self.dat_beam3D = ct.GradButton(
            self, bmp=beamXY_bmp, label="  Beam XYZ Plot", size=(250, -1)
        )
        dat_btn_sizer.Add(self.dat_reshist)
        dat_btn_sizer.Add(self.dat_beamxy, flag=wx.TOP, border=5)
        dat_btn_sizer.Add(self.dat_beam3D, flag=wx.TOP, border=5)
        self.Bind(wx.EVT_BUTTON, self.onPlotBeamXY, self.dat_beamxy)
        self.Bind(wx.EVT_BUTTON, self.onPlotBeam3D, self.dat_beam3D)
        self.Bind(wx.EVT_BUTTON, self.onPlotResHist, self.dat_reshist)

        # Insert into sizers
        dat_box_sizer.Add(dat_box_grid, flag=wx.ALL, border=10)
        dat_box_sizer.AddStretchSpacer()
        dat_box_sizer.Add(dat_btn_sizer, flag=wx.ALL, border=10)
        summary_sizer.Add(dat_box_sizer, flag=wx.EXPAND | wx.ALL, border=10)

        # # Summary
        smr_box = wx.StaticBox(self, label="Run Summary")
        smr_box.SetFont(sfont)
        smr_box_sizer = wx.StaticBoxSizer(smr_box, wx.HORIZONTAL)
        smr_box_grid = wx.FlexGridSizer(7, 2, 5, 20)
        smr_btn_sizer = wx.BoxSizer(wx.VERTICAL)

        self.readin_txt = wx.StaticText(self, label="250")
        self.readin_txt.SetFont(sfont)
        self.nodiff_txt = wx.StaticText(self, label="100")
        self.nodiff_txt.SetFont(sfont)
        self.w_diff_txt = wx.StaticText(self, label="150")
        self.w_diff_txt.SetFont(sfont)
        self.noint_txt = wx.StaticText(self, label="30")
        self.noint_txt.SetFont(sfont)
        self.final_txt = wx.StaticText(self, label="100")
        self.final_txt.SetFont(sfont)

        smr_box_grid.AddMany(
            [
                (wx.StaticText(self, label="Read in: ")),
                (self.readin_txt),
                (wx.StaticText(self, label="No diffraction:")),
                (self.nodiff_txt),
                (wx.StaticText(self, label="Have diffraction: ")),
                (self.w_diff_txt),
            ]
        )

        prime_bmp = bitmaps.fetch_custom_icon_bitmap("prime32", scale=(24, 24))
        self.smr_runprime = ct.GradButton(
            self, bmp=prime_bmp, label="  Run PRIME", size=(250, -1)
        )

        smr_btn_sizer.Add(self.smr_runprime)
        # smr_btn_sizer.Add(self.smr_runmerge, flag=wx.TOP, border=5)
        self.Bind(wx.EVT_BUTTON, self.onPRIME, self.smr_runprime)

        if self.gparams.advanced.integrate_with == "cctbx":
            self.noprf_txt = wx.StaticText(self, label="20")
            self.noprf_txt.SetFont(sfont)
            smr_box_grid.AddMany(
                [
                    (wx.StaticText(self, label="Failed indexing / integration")),
                    (self.noint_txt),
                    (wx.StaticText(self, label="Failed filter")),
                    (self.noprf_txt),
                ]
            )
        elif self.gparams.advanced.integrate_with == "dials":
            self.nospf_txt = wx.StaticText(self, label="10")
            self.nospf_txt.SetFont(sfont)
            self.noidx_txt = wx.StaticText(self, label="20")
            self.noidx_txt.SetFont(sfont)
            smr_box_grid.AddMany(
                [
                    (wx.StaticText(self, label="Failed spotfinding")),
                    (self.nospf_txt),
                    (wx.StaticText(self, label="Failed indexing")),
                    (self.noidx_txt),
                    (wx.StaticText(self, label="Failed integration")),
                    (self.noint_txt),
                ]
            )
        smr_box_grid.AddMany(
            [(wx.StaticText(self, label="Final integrated pickles")), (self.final_txt)]
        )

        smr_box_sizer.Add(smr_box_grid, flag=wx.ALL, border=10)
        smr_box_sizer.AddStretchSpacer()
        smr_box_sizer.Add(smr_btn_sizer, flag=wx.ALL, border=10)
        summary_sizer.Add(smr_box_sizer, flag=wx.EXPAND | wx.ALL, border=10)

        self.SetFont(sfont)
        self.SetSizer(summary_sizer)

    def onPRIME(self, e):
        from prime.postrefine.mod_gui_init import PRIMEWindow

        self.prime_window = PRIMEWindow(
            self, -1, title="PRIME", prefix=self.gparams.advanced.prime_prefix
        )
        self.prime_window.load_script(out_dir=self.out_dir)
        self.prime_window.SetMinSize(self.prime_window.GetEffectiveMinSize())
        self.prime_window.Show(True)

    def onPlotHeatmap(self, e):
        if self.final_objects != None:
            self.plot.plot_spotfinding_heatmap()

    def onPlotBeamXY(self, e):
        if self.final_objects != None:
            self.plot.plot_beam_xy()

    def onPlotBeam3D(self, e):
        if self.final_objects != None:
            self.plot.plot_beam_xy(threeD=True)

    def onPlotResHist(self, e):
        if self.final_objects != None:
            self.plot.plot_res_histogram()


class ProcWindow(wx.Frame):
    """New frame that will show processing info."""

    def __init__(self, parent, id, title, phil, target_phil=None, test=False):
        wx.Frame.__init__(
            self,
            parent,
            id,
            title,
            size=(800, 900),
            style=wx.SYSTEM_MENU | wx.CAPTION | wx.CLOSE_BOX | wx.RESIZE_BORDER,
        )

        self.logtext = ""
        self.finished_objects = []
        self.read_object_files = []
        self.obj_counter = 0
        self.bookmark = 0
        self.gparams = phil.extract()
        self.target_phil = target_phil
        self.state = "process"
        self.monitor_mode = False
        self.monitor_mode_timeout = None
        self.timeout_start = None
        self.new_images = []
        self.find_new_images = True

        self.main_panel = wx.Panel(self)
        self.main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Toolbar
        self.proc_toolbar = self.CreateToolBar(style=wx.TB_3DBUTTONS | wx.TB_TEXT)
        abort_bmp = bitmaps.fetch_icon_bitmap("actions", "stop")
        self.tb_btn_abort = self.proc_toolbar.AddLabelTool(
            wx.ID_ANY, label="Abort", bitmap=abort_bmp, shortHelp="Abort"
        )
        resume_bmp = bitmaps.fetch_icon_bitmap("actions", "quick_restart")
        self.tb_btn_resume = self.proc_toolbar.AddLabelTool(
            wx.ID_ANY, label="Resume", bitmap=resume_bmp, shortHelp="Resume aborted run"
        )
        self.proc_toolbar.EnableTool(self.tb_btn_resume.GetId(), False)
        self.proc_toolbar.AddSeparator()
        watch_bmp = bitmaps.fetch_icon_bitmap("apps", "search")
        self.tb_btn_monitor = self.proc_toolbar.AddCheckLabelTool(
            wx.ID_ANY, label="Monitor", bitmap=watch_bmp, shortHelp="Monitor Mode"
        )
        self.proc_toolbar.Realize()

        # Status box
        self.status_panel = wx.Panel(self.main_panel)
        self.status_sizer = wx.BoxSizer(wx.VERTICAL)
        self.status_box = wx.StaticBox(self.status_panel, label="Status")
        self.status_box_sizer = wx.StaticBoxSizer(self.status_box, wx.HORIZONTAL)
        self.status_txt = wx.StaticText(self.status_panel, label="")
        self.status_box_sizer.Add(
            self.status_txt, flag=wx.ALL | wx.ALIGN_CENTER, border=10
        )
        self.status_sizer.Add(self.status_box_sizer, flag=wx.EXPAND | wx.ALL, border=3)
        self.status_panel.SetSizer(self.status_sizer)

        # Tabbed output window(s)
        self.proc_panel = wx.Panel(self.main_panel)
        self.proc_nb = wx.Notebook(self.proc_panel, style=0)
        self.chart_tab = ProcessingTab(self.proc_nb)
        self.log_tab = LogTab(self.proc_nb)
        self.proc_nb.AddPage(self.log_tab, "Log")
        self.proc_nb.AddPage(self.chart_tab, "Charts")
        self.proc_nb.SetSelection(1)
        self.proc_sizer = wx.BoxSizer(wx.VERTICAL)
        self.proc_sizer.Add(self.proc_nb, 1, flag=wx.EXPAND | wx.ALL, border=3)
        self.proc_panel.SetSizer(self.proc_sizer)

        self.main_sizer.Add(self.status_panel, flag=wx.EXPAND | wx.ALL, border=3)
        self.main_sizer.Add(self.proc_panel, 1, flag=wx.EXPAND | wx.ALL, border=3)
        self.main_panel.SetSizer(self.main_sizer)

        # Processing status bar
        self.sb = self.CreateStatusBar()
        self.sb.SetFieldsCount(2)
        self.sb.SetStatusWidths([-1, -2])

        # Output gauge in status bar
        self.gauge_process = wx.Gauge(
            self.sb, -1, style=wx.GA_HORIZONTAL | wx.GA_SMOOTH
        )
        rect = self.sb.GetFieldRect(0)
        self.gauge_process.SetPosition((rect.x + 2, rect.y + 2))
        self.gauge_process.SetSize((rect.width - 4, rect.height - 4))
        self.gauge_process.Hide()

        # Output polling timer
        self.timer = wx.Timer(self)

        # Event bindings
        self.Bind(thr.EVT_ALLDONE, self.onFinishedProcess)
        self.Bind(thr.EVT_IMGDONE, self.onFinishedImageFinder)
        self.sb.Bind(wx.EVT_SIZE, self.onStatusBarResize)
        self.Bind(wx.EVT_TIMER, self.onTimer, id=self.timer.GetId())

        # Button bindings
        self.Bind(wx.EVT_TOOL, self.onAbort, self.tb_btn_abort)
        self.Bind(wx.EVT_TOOL, self.onResume, self.tb_btn_resume)
        self.Bind(wx.EVT_TOOL, self.onMonitor, self.tb_btn_monitor)

        # Determine if monitor mode was previously selected
        if self.gparams.advanced.monitor_mode:
            self.proc_toolbar.ToggleTool(self.tb_btn_monitor.GetId(), True)
            self.monitor_mode = True
            if self.gparams.advanced.monitor_mode_timeout:
                if self.gparams.advanced.monitor_mode_timeout_length is None:
                    self.monitor_mode_timeout = 30
                else:
                    self.monitor_mode_timeout = (
                        self.gparams.advanced.monitor_mode_timeout_length
                    )

    def onMonitor(self, e):
        if self.proc_toolbar.GetToolState(self.tb_btn_monitor.GetId()):
            self.monitor_mode = True
            if self.gparams.advanced.monitor_mode_timeout:
                if self.gparams.advanced.monitor_mode_timeout_length is None:
                    self.monitor_mode_timeout = 30
                else:
                    self.monitor_mode_timeout = (
                        self.gparams.advanced.monitor_mode_timeout_length
                    )
        elif not self.proc_toolbar.GetToolState(self.tb_btn_monitor.GetId()):
            self.monitor_mode = False
            self.monitor_mode_timeout = None

    def onStatusBarResize(self, e):
        rect = self.sb.GetFieldRect(0)
        self.gauge_process.SetPosition((rect.x + 2, rect.y + 2))
        self.gauge_process.SetSize((rect.width - 4, rect.height - 4))
        self.Refresh()

    def onAbort(self, e):
        with open(self.tmp_abort_file, "w") as af:
            af.write("")
        self.status_txt.SetForegroundColour("red")
        self.status_txt.SetLabel("Aborting...")
        self.proc_toolbar.EnableTool(self.tb_btn_abort.GetId(), False)

    def onResume(self, e):
        """Restarts an aborted run if the processing window is still open.

        Basically goes through self.finished_objects, extracts the raw
        image names and regenerates the self.img_list to only have those
        image paths; then finds any 'new' images (which includes
        unprocessed images as well as any images that may have been
        added during the abort pause) and runs processing
        """

        # Remove abort signal file
        os.remove(self.tmp_abort_file)

        # Re-generate new image info to include un-processed images
        ginp = GenerateInput()
        input_entries = [i for i in self.gparams.input if i != None]
        ext_file_list = ginp.make_input_list(input_entries)
        old_file_list = [i.raw_img for i in self.finished_objects]
        new_file_list = [i for i in ext_file_list if i not in old_file_list]

        # Generate list of new images
        self.new_images = [
            [i, len(ext_file_list) + 1, j]
            for i, j in enumerate(new_file_list, len(old_file_list) + 1)
        ]

        # Reset self.img_list to only processed images
        self.img_list = [
            [i, len(ext_file_list) + 1, j] for i, j in enumerate(old_file_list, 1)
        ]

        # # Re-initialize monitor mode
        # if self.monitor_mode:
        #   self.new_images = []
        #   self.find_new_images = True
        #   if self.monitor_mode_timeout:
        #     self.timeout_start = None

        # Reset toolbar buttons
        self.proc_toolbar.EnableTool(self.tb_btn_abort.GetId(), True)
        self.proc_toolbar.EnableTool(self.tb_btn_resume.GetId(), False)
        self.proc_toolbar.EnableTool(self.tb_btn_monitor.GetId(), True)

        # Run processing, etc.
        self.state = "resume"
        self.process_images()
        self.timer.Start(5000)

    def run(self, init):
        # Initialize IOTA parameters and log
        self.init = init
        good_init = self.init.run(self.gparams, target_phil=self.target_phil)

        # Start process
        if good_init:
            self.tmp_abort_file = os.path.join(self.init.int_base, ".abort.tmp")
            self.status_txt.SetForegroundColour("black")
            self.status_txt.SetLabel("Running...")
            self.process_images()
            self.good_to_go = True
            self.timer.Start(5000)

            # write init file
            ep.dump(os.path.join(self.gparams.output, "init.cfg"), self.init)

        else:
            self.good_to_go = False

    def process_images(self):
        """One-fell-swoop importing / triaging / integration of images."""

        # Set font properties for status window
        font = self.sb.GetFont()
        font.SetWeight(wx.NORMAL)
        self.status_txt.SetFont(font)
        self.status_txt.SetForegroundColour("black")

        if self.init.params.cctbx.selection.select_only.flag_on:
            self.img_list = [
                [i, len(self.init.gs_img_objects) + 1, j]
                for i, j in enumerate(self.init.gs_img_objects, 1)
            ]
            iterable = self.img_list
            type = "object"
            self.status_txt.SetLabel("Re-running selection...")
        else:
            type = "image"
            if self.state == "process":
                self.img_list = [
                    [i, len(self.init.input_list) + 1, j]
                    for i, j in enumerate(self.init.input_list, 1)
                ]
                iterable = self.img_list
                self.status_summary = [0] * len(self.img_list)
                self.nref_list = [0] * len(self.img_list)
                self.nref_xaxis = [i[0] for i in self.img_list]
                self.res_list = [0] * len(self.img_list)
                self.status_txt.SetLabel(
                    "Processing {} images..." "".format(len(self.img_list))
                )
            elif self.state == "new images":
                iterable = self.new_images
                self.img_list.extend(self.new_images)
                self.new_images = []
                self.status_summary.extend([0] * len(iterable))
                self.nref_list.extend([0] * len(iterable))
                self.nref_xaxis.extend([i[0] for i in iterable])
                self.res_list.extend([0] * len(iterable))
                self.status_txt.SetForegroundColour("black")
                self.status_txt.SetLabel(
                    "Processing additional {} images ({} total)..."
                    "".format(len(iterable), len(self.img_list))
                )
                self.plot_integration()
            elif self.state == "resume":
                iterable = self.new_images
                self.img_list.extend(self.new_images)
                self.new_images = []
                self.status_summary = [0] * len(self.img_list)
                self.nref_list = [0] * len(self.img_list)
                self.nref_xaxis = [i[0] for i in self.img_list]
                self.res_list = [0] * len(self.img_list)
                self.status_txt.SetLabel(
                    "Processing {} remaining images ({} total)..."
                    "".format(len(iterable), len(self.img_list))
                )

        self.gauge_process.SetRange(len(self.img_list))
        img_process = thr.ProcThread(self, self.init, iterable, input_type=type)
        img_process.start()

    def analyze_results(self):
        if len(self.final_objects) == 0:
            self.display_log()
            self.plot_integration()
            self.status_txt.SetForegroundColor("red")
            self.status_txt.SetLabel("No images successfully integrated")

        elif not self.gparams.image_conversion.convert_only:
            self.status_txt.SetForegroundColour("black")
            self.status_txt.SetLabel("Analyzing results...")

            # Do analysis
            analysis = Analyzer(self.init, self.finished_objects, gui_mode=True)
            plot = Plotter(self.gparams, self.final_objects, self.init.viz_base)

            # Initialize summary tab
            prime_file = os.path.join(
                self.init.int_base, "{}.phil".format(self.gparams.advanced.prime_prefix)
            )
            self.summary_tab = SummaryTab(
                self.proc_nb,
                self.gparams,
                self.final_objects,
                os.path.dirname(prime_file),
                plot,
            )

            # Run information
            self.summary_tab.title_txt.SetLabel(noneset(self.gparams.description))
            self.summary_tab.folder_txt.SetLabel(self.gparams.output)

            # Analysis of integration
            if self.gparams.advanced.integrate_with == "cctbx":
                self.summary_tab.sih_min.SetLabel("{:4.0f}".format(np.min(analysis.s)))
                self.summary_tab.sih_max.SetLabel("{:4.0f}".format(np.max(analysis.s)))
                self.summary_tab.sih_avg.SetLabel("{:4.2f}".format(np.mean(analysis.s)))
                self.summary_tab.sih_std.SetLabel("{:4.2f}".format(np.std(analysis.s)))
                self.summary_tab.sph_min.SetLabel("{:4.0f}".format(np.min(analysis.h)))
                self.summary_tab.sph_max.SetLabel("{:4.0f}".format(np.max(analysis.h)))
                self.summary_tab.sph_avg.SetLabel("{:4.2f}".format(np.mean(analysis.h)))
                self.summary_tab.sph_std.SetLabel("{:4.2f}".format(np.std(analysis.h)))
                self.summary_tab.spa_min.SetLabel("{:4.0f}".format(np.min(analysis.a)))
                self.summary_tab.spa_max.SetLabel("{:4.0f}".format(np.max(analysis.a)))
                self.summary_tab.spa_avg.SetLabel("{:4.2f}".format(np.mean(analysis.a)))
                self.summary_tab.spa_std.SetLabel("{:4.2f}".format(np.std(analysis.a)))

            # Dataset information
            analysis.print_results()
            pg, uc = analysis.unit_cell_analysis()

            self.summary_tab.pg_txt.SetLabel(str(pg))
            unit_cell = " ".join(["{:4.1f}".format(i) for i in uc])
            self.summary_tab.uc_txt.SetLabel(unit_cell)
            res = to_unicode(
                u"{:4.2f} - {:4.2f} {}".format(
                    np.mean(analysis.lres), np.mean(analysis.hres), u"\u212B"
                )
            )
            self.summary_tab.rs_txt.SetLabel(res)

            with warnings.catch_warnings():
                # To catch any 'mean of empty slice' runtime warnings
                warnings.simplefilter("ignore", category=RuntimeWarning)
                beamX, beamY = plot.calculate_beam_xy()[:2]
                beamXY = "X = {:4.1f} mm, Y = {:4.1f} mm" "".format(
                    np.median(beamX), np.median(beamY)
                )

            self.summary_tab.xy_txt.SetLabel(beamXY)

            # Summary
            self.summary_tab.readin_txt.SetLabel(str(len(analysis.all_objects)))
            self.summary_tab.nodiff_txt.SetLabel(str(len(analysis.no_diff_objects)))
            self.summary_tab.w_diff_txt.SetLabel(str(len(analysis.diff_objects)))
            if self.gparams.advanced.integrate_with == "cctbx":
                self.summary_tab.noint_txt.SetLabel(str(len(analysis.not_int_objects)))
                self.summary_tab.noprf_txt.SetLabel(
                    str(len(analysis.filter_fail_objects))
                )
            elif self.gparams.advanced.integrate_with == "dials":
                self.summary_tab.nospf_txt.SetLabel(str(len(analysis.not_spf_objects)))
                self.summary_tab.noidx_txt.SetLabel(str(len(analysis.not_idx_objects)))
                self.summary_tab.noint_txt.SetLabel(str(len(analysis.not_int_objects)))
            self.summary_tab.final_txt.SetLabel(str(len(analysis.final_objects)))

            # Generate input file for PRIME
            analysis.print_summary()
            analysis.make_prime_input(filename=prime_file)

            # Display summary
            self.proc_nb.AddPage(self.summary_tab, "Analysis")
            self.proc_nb.SetSelection(2)

            # Signal end of run
            font = self.sb.GetFont()
            font.SetWeight(wx.BOLD)
            self.status_txt.SetFont(font)
            self.status_txt.SetForegroundColour("blue")
            self.status_txt.SetLabel("DONE")

            # Finish up
            self.display_log()
            self.plot_integration()

        # Stop timer
        self.timer.Stop()

    def display_log(self):
        """Display PRIME stdout."""
        if os.path.isfile(self.init.logfile):
            with open(self.init.logfile, "r") as out:
                out.seek(self.bookmark)
                output = out.readlines()
                self.bookmark = out.tell()

            ins_pt = self.log_tab.log_window.GetInsertionPoint()
            for i in output:
                self.log_tab.log_window.AppendText(i)
                self.log_tab.log_window.SetInsertionPoint(ins_pt)

    def plot_integration(self):
        try:
            # Summary pie chart
            categories = [
                [
                    "failed triage",
                    "#d73027",
                    len(
                        [i for i in self.finished_objects if i.fail == "failed triage"]
                    ),
                ],
                [
                    "failed indexing / integration",
                    "#f46d43",
                    len(
                        [
                            i
                            for i in self.finished_objects
                            if i.fail == "failed grid search"
                        ]
                    ),
                ],
                [
                    "failed filter",
                    "#ffffbf",
                    len(
                        [
                            i
                            for i in self.finished_objects
                            if i.fail == "failed prefilter"
                        ]
                    ),
                ],
                [
                    "failed spotfinding",
                    "#f46d43",
                    len(
                        [
                            i
                            for i in self.finished_objects
                            if i.fail == "failed spotfinding"
                        ]
                    ),
                ],
                [
                    "failed indexing",
                    "#fdae61",
                    len(
                        [
                            i
                            for i in self.finished_objects
                            if i.fail == "failed indexing"
                        ]
                    ),
                ],
                [
                    "failed integration",
                    "#fee090",
                    len(
                        [
                            i
                            for i in self.finished_objects
                            if i.fail == "failed integration"
                        ]
                    ),
                ],
                [
                    "integrated",
                    "#4575b4",
                    len(
                        [
                            i
                            for i in self.finished_objects
                            if i.fail == None and i.final["final"] != None
                        ]
                    ),
                ],
            ]
            categories.append(
                [
                    "not processed",
                    "#e0f3f8",
                    len(self.img_list) - sum([i[2] for i in categories]),
                ]
            )
            names = [i[0] for i in categories if i[2] > 0]
            colors = [i[1] for i in categories if i[2] > 0]
            numbers = [i[2] for i in categories if i[2] > 0]

            self.chart_tab.int_axes.clear()
            self.chart_tab.int_axes.pie(numbers, autopct="%.0f%%", colors=colors)
            self.chart_tab.int_axes.legend(
                names, loc="lower left", fontsize=9, fancybox=True
            )
            self.chart_tab.int_axes.axis("equal")

        except ValueError, e:
            pass

        if sum(self.nref_list) > 0 and sum(self.res_list) > 0:
            try:
                # Strong reflections per frame
                self.chart_tab.nsref_axes.clear()
                nsref_x = np.array([i + 1.5 for i in range(len(self.img_list))]).astype(
                    np.double
                )
                nsref_y = np.array(
                    [np.nan if i == 0 else i for i in self.nref_list]
                ).astype(np.double)
                nsref_ylabel = "Reflections (I/{0}(I) > {1})" "".format(
                    r"$\sigma$", self.gparams.cctbx.selection.min_sigma
                )
                nsref = self.chart_tab.nsref_axes.scatter(
                    nsref_x,
                    nsref_y,
                    s=45,
                    marker="o",
                    edgecolors="black",
                    color="#ca0020",
                    picker=True,
                )

                nsref_median = np.median([i for i in self.nref_list if i > 0])
                nsref_med = self.chart_tab.nsref_axes.axhline(
                    nsref_median, c="#ca0020", ls="--"
                )

                self.chart_tab.nsref_axes.set_xlim(0, np.nanmax(nsref_x) + 2)
                nsref_ymax = np.nanmax(nsref_y) * 1.25 + 10
                if nsref_ymax == 0:
                    nsref_ymax = 100
                self.chart_tab.nsref_axes.set_ylim(ymin=0, ymax=nsref_ymax)
                self.chart_tab.nsref_axes.set_ylabel(nsref_ylabel, fontsize=11)
                self.chart_tab.nsref_axes.set_xlabel("Frame")
                self.chart_tab.nsref_axes.yaxis.get_major_ticks()[0].label1.set_visible(
                    False
                )
                self.chart_tab.nsref_axes.yaxis.get_major_ticks()[
                    -1
                ].label1.set_visible(False)

                # Resolution per frame
                self.chart_tab.res_axes.clear()
                res_x = np.array([i + 1.5 for i in range(len(self.img_list))]).astype(
                    np.double
                )
                res_y = np.array(
                    [np.nan if i == 0 else i for i in self.res_list]
                ).astype(np.double)
                res_m = np.isfinite(res_y)

                res = self.chart_tab.res_axes.scatter(
                    res_x[res_m],
                    res_y[res_m],
                    s=45,
                    marker="o",
                    edgecolors="black",
                    color="#0571b0",
                    picker=True,
                )
                res_median = np.median([i for i in self.res_list if i > 0])
                res_med = self.chart_tab.res_axes.axhline(
                    res_median, c="#0571b0", ls="--"
                )

                self.chart_tab.res_axes.set_xlim(0, np.nanmax(res_x) + 2)
                res_ymax = np.nanmax(res_y) * 1.1
                res_ymin = np.nanmin(res_y) * 0.9
                if res_ymin == res_ymax:
                    res_ymax = res_ymin + 1
                self.chart_tab.res_axes.set_ylim(ymin=res_ymin, ymax=res_ymax)
                res_ylabel = "Resolution ({})".format(r"$\AA$")
                self.chart_tab.res_axes.set_ylabel(res_ylabel, fontsize=11)
                self.chart_tab.res_axes.yaxis.get_major_ticks()[0].label1.set_visible(
                    False
                )
                self.chart_tab.res_axes.yaxis.get_major_ticks()[-1].label1.set_visible(
                    False
                )
                plt.setp(self.chart_tab.res_axes.get_xticklabels(), visible=False)

            except ValueError, e:
                pass

            try:
                # Beam XY (cumulative)
                info = []
                wavelengths = []
                distances = []
                cells = []

                # Import relevant info
                for root, dirs, files in os.walk(self.init.fin_base):
                    for filename in files:
                        found_file = os.path.join(root, filename)
                        if found_file.endswith(("pickle")):
                            beam = ep.load(found_file)
                            info.append([found_file, beam["xbeam"], beam["ybeam"]])
                            wavelengths.append(beam["wavelength"])
                            distances.append(beam["distance"])
                            cells.append(
                                beam["observations"][0].unit_cell().parameters()
                            )

                # Calculate beam center coordinates and distances
                if len(info) > 0:
                    beamX = [i[1] for i in info]
                    beamY = [j[2] for j in info]
                    beam_dist = [
                        math.hypot(i[1] - np.median(beamX), i[2] - np.median(beamY))
                        for i in info
                    ]

                    wavelength = np.median(wavelengths)
                    det_distance = np.median(distances)
                    a = np.median([i[0] for i in cells])
                    b = np.median([i[1] for i in cells])
                    c = np.median([i[2] for i in cells])

                    # Calculate predicted L +/- 1 misindexing distance for each cell edge
                    aD = det_distance * math.tan(2 * math.asin(wavelength / (2 * a)))
                    bD = det_distance * math.tan(2 * math.asin(wavelength / (2 * b)))
                    cD = det_distance * math.tan(2 * math.asin(wavelength / (2 * c)))

                    # Calculate axis limits of beam center scatter plot
                    beamxy_delta = np.ceil(np.max(beam_dist))
                    xmax = round(np.median(beamX) + beamxy_delta)
                    xmin = round(np.median(beamX) - beamxy_delta)
                    ymax = round(np.median(beamY) + beamxy_delta)
                    ymin = round(np.median(beamY) - beamxy_delta)

                    if xmax == xmin:
                        xmax += 0.5
                        xmin -= 0.5
                    if ymax == ymin:
                        ymax += 0.5
                        ymin -= 0.5

                    # Plot beam center scatter plot
                    self.chart_tab.bxy_axes.clear()
                    self.chart_tab.bxy_axes.axis("equal")
                    self.chart_tab.bxy_axes.axis([xmin, xmax, ymin, ymax])
                    self.chart_tab.bxy_axes.scatter(
                        beamX, beamY, alpha=1, s=20, c="grey", lw=1
                    )
                    self.chart_tab.bxy_axes.plot(
                        np.median(beamX),
                        np.median(beamY),
                        markersize=8,
                        marker="o",
                        c="yellow",
                        lw=2,
                    )

                    # Plot projected mis-indexing limits for all three axes
                    circle_a = plt.Circle(
                        (np.median(beamX), np.median(beamY)),
                        radius=aD,
                        color="r",
                        fill=False,
                        clip_on=True,
                    )
                    circle_b = plt.Circle(
                        (np.median(beamX), np.median(beamY)),
                        radius=bD,
                        color="g",
                        fill=False,
                        clip_on=True,
                    )
                    circle_c = plt.Circle(
                        (np.median(beamX), np.median(beamY)),
                        radius=cD,
                        color="b",
                        fill=False,
                        clip_on=True,
                    )
                    self.chart_tab.bxy_axes.add_patch(circle_a)
                    self.chart_tab.bxy_axes.add_patch(circle_b)
                    self.chart_tab.bxy_axes.add_patch(circle_c)
                    self.chart_tab.bxy_axes.set_xlabel("BeamX (mm)", fontsize=12)
                    self.chart_tab.bxy_axes.set_ylabel("BeamY (mm)", fontsize=12)
                    self.chart_tab.bxy_axes.set_title("Beam Center Coordinates")

            except ValueError, e:
                pass

        self.chart_tab.canvas.draw()
        self.chart_tab.Layout()

    def onTimer(self, e):
        if os.path.isfile(self.tmp_abort_file):
            self.finish_process()

        img_object_files = [
            os.path.join(self.init.obj_base, i)
            for i in os.listdir(self.init.obj_base)
            if i.endswith("int")
        ]
        new_objects = [
            ep.load(i) for i in img_object_files if i not in self.read_object_files
        ]
        self.finished_objects.extend([i for i in new_objects if i.status == "final"])
        self.read_object_files = [i.obj_file for i in self.finished_objects]

        if len(self.finished_objects) > 0:
            for obj in self.finished_objects:
                self.nref_list[obj.img_index - 1] = obj.final["strong"]
                self.res_list[obj.img_index - 1] = obj.final["res"]

        if len(self.finished_objects) > self.obj_counter:
            self.plot_integration()
            self.obj_counter = len(self.finished_objects)

        # Update gauge
        self.gauge_process.Show()
        self.gauge_process.SetValue(len(self.finished_objects))

        # Update status bar
        if self.gparams.image_conversion.convert_only:
            img_with_diffraction = [
                i
                for i in self.finished_objects
                if i.status == "imported" and i.fail == None
            ]
            self.sb.SetStatusText(
                "{} of {} images imported, {} have diffraction"
                "".format(
                    self.obj_counter,
                    len(self.init.input_list),
                    len(img_with_diffraction),
                ),
                1,
            )
        else:
            processed_images = [i for i in self.finished_objects if i.fail == None]
            self.sb.SetStatusText(
                "{} of {} images processed, {} successfully integrated"
                "".format(self.obj_counter, len(self.img_list), len(processed_images)),
                1,
            )

        # Update log
        self.display_log()

        # Run an instance of new image finder on a separate thread
        if self.find_new_images:
            self.find_new_images = False
            ext_image_list = self.img_list + self.new_images
            img_finder = thr.ImageFinderThread(
                self, image_paths=self.gparams.input, image_list=ext_image_list
            )
            img_finder.start()

        # Check if all images have been looked at; if yes, finish process
        if self.obj_counter >= len(self.img_list):
            if self.monitor_mode:
                if len(self.new_images) > 0:
                    self.timeout_start = None
                    self.state = "new images"
                    self.process_images()
                else:
                    if self.monitor_mode_timeout != None:
                        if self.timeout_start is None:
                            self.timeout_start = time.time()
                        else:
                            interval = time.time() - self.timeout_start
                            if interval >= self.monitor_mode_timeout:
                                self.status_txt.SetLabel("Timed out. Finishing...")
                                self.finish_process()
                            else:
                                timeout_msg = (
                                    "No images found! Timing out in {} seconds"
                                    "".format(int(self.monitor_mode_timeout - interval))
                                )
                                self.status_txt.SetLabel(timeout_msg)
            else:
                self.finish_process()

    def onFinishedProcess(self, e):
        pass
        # if self.gparams.mp_method != 'lsf':
        #   self.img_objects = e.GetValue()
        #   self.finish_process()

    def onFinishedImageFinder(self, e):
        self.new_images = self.new_images + e.GetValue()
        self.find_new_images = True

    def finish_process(self):
        import shutil

        if os.path.isfile(self.tmp_abort_file):
            self.gauge_process.Hide()
            font = self.sb.GetFont()
            font.SetWeight(wx.BOLD)
            self.status_txt.SetFont(font)
            self.status_txt.SetForegroundColour("red")
            self.status_txt.SetLabel("ABORTED BY USER")
            self.timer.Stop()
            self.proc_toolbar.EnableTool(self.tb_btn_resume.GetId(), True)
            shutil.rmtree(self.init.tmp_base)
            return
        else:
            self.final_objects = [i for i in self.finished_objects if i.fail == None]
            self.gauge_process.Hide()
            self.proc_toolbar.EnableTool(self.tb_btn_abort.GetId(), False)
            self.proc_toolbar.EnableTool(self.tb_btn_monitor.GetId(), False)
            self.proc_toolbar.ToggleTool(self.tb_btn_monitor.GetId(), False)
            self.sb.SetStatusText(
                "{} of {} images successfully integrated"
                "".format(len(self.final_objects), len(self.img_list)),
                1,
            )
            if len(self.final_objects) > 0:
                self.plot_integration()
                self.analyze_results()

            shutil.rmtree(self.init.tmp_base)