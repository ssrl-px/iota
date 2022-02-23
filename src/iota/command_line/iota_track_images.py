from __future__ import absolute_import, division, print_function

"""
Author      : Lyubimov, A.Y.
Created     : 07/21/2017
Last Changed: 08/29/2018
Description : IOTA image-tracking GUI launcher
"""

import wx
from iota import intx_version
from iota.etc import iota_tracker as trk


class MainApp(wx.App):
    """App for the main GUI window."""

    def OnInit(self):
        self.frame = trk.TrackerWindow(
            None, -1, title="INTERCEPTOR v.{}" "".format(intx_version)
        )
        self.frame.SetMinSize(self.frame.GetEffectiveMinSize())
        self.frame.SetPosition((150, 150))
        self.frame.Show(True)
        self.frame.Layout()
        self.SetTopWindow(self.frame)
        return True


# ---------------------------------------------------------------------------- #


def entry_point():
    app = MainApp(0)
    app.MainLoop()


if __name__ == "__main__":
    entry_point()
