# -*- coding: utf-8 -*-

#
# WX example for VLC Python bindings
# Copyright (C) 2009-2010 the VideoLAN team
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
#
"""
A modified version of the simple example for VLC python bindings using wxPython.

@author: Michele Orrù
@author: Gary feng
@author: Dan Blanchard
@date: November 2010
"""


# import standard libraries
import os
import re
import sys
import subprocess
import logging

# import external libraries
import numpy as np
import vlc
import wx  # 2.8
import win32gui  # http://stackoverflow.com/questions/2090464/python-window-activation
from ScreenCaster import ScreenCaster


class WindowMgr:
    """Encapsulates some calls to the winapi for window management"""
    def __init__(self):
        """Constructor"""
        self._handle = None

    def find_window(self, class_name, window_name=None):
        """find a window by its class_name"""
        self._handle = win32gui.FindWindow(class_name, window_name)

    def _window_enum_callback(self, hwnd, wildcard):
        '''Pass to win32gui.EnumWindows() to check all the opened windows'''
        if re.match(wildcard, str(win32gui.GetWindowText(hwnd))) is not None:
            self._handle = hwnd

    def find_window_wildcard(self, wildcard):
        self._handle = None
        win32gui.EnumWindows(self._window_enum_callback, wildcard)

    def set_foreground(self):
        """put the window in the foreground"""
        win32gui.SetForegroundWindow(self._handle)

w = WindowMgr()
# w.find_window_wildcard(".*Hello.*")
# w.set_foreground()


class Player(wx.Frame):
    # need it to stop timer from updating the slider when it's dragged
    bSliderDragged = False

    basename = ""
    basedir = ""
    logger = None

    timerInUse = False

    def __init__(self, title):
        wx.Frame.__init__(self, None, -1, title,
                          pos=wx.DefaultPosition, size=(500, 150))

        # Menu Bar
        #   File Menu
        self.frame_menubar = wx.MenuBar()
        self.file_menu = wx.Menu()
        self.file_menu.Append(1, "&Open", "Open from file..")
        self.file_menu.AppendSeparator()
        self.file_menu.Append(2, "&Close", "Quit")
        self.Bind(wx.EVT_MENU, self.OnOpen, id=1)
        self.Bind(wx.EVT_MENU, self.OnExit, id=2)
        self.frame_menubar.Append(self.file_menu, "File")
        self.SetMenuBar(self.frame_menubar)

        # Panels
        # The first panel holds the video and it's all black
        self.videopanel = wx.Panel(self, -1)
        self.videopanel.SetBackgroundColour(wx.BLACK)

        # The second panel holds controls
        ctrlpanel = wx.Panel(self, -1)
        self.timeslider = wx.Slider(ctrlpanel, -1, 0, 0, 1000)
        self.timeslider.SetRange(0, 1000)
        pause = wx.Button(ctrlpanel, label="Pause")
        play = wx.Button(ctrlpanel, label="Play")
        stop = wx.Button(ctrlpanel, label="Stop")
        ff = wx.Button(ctrlpanel, label=">>")
        rev = wx.Button(ctrlpanel, label="<<")
        volume = wx.Button(ctrlpanel, label="Volume")
        self.volslider = wx.Slider(ctrlpanel, -1, 0, 0, 100, size=(100, -1))

        # Bind controls to events
        self.Bind(wx.EVT_BUTTON, self.OnPlay, play)
        self.Bind(wx.EVT_BUTTON, self.OnPause, pause)
        self.Bind(wx.EVT_BUTTON, self.OnStop, stop)
        self.Bind(wx.EVT_BUTTON, self.OnLastPage, rev)
        self.Bind(wx.EVT_BUTTON, self.OnNextPage, ff)
        self.Bind(wx.EVT_BUTTON, self.OnToggleVolume, volume)
        self.Bind(wx.EVT_SLIDER, self.OnSetVolume, self.volslider)
        # respond to changes in slider position ...
        self.Bind(wx.EVT_SLIDER, self.OnAdjust, self.timeslider)

        # Give a pretty layout to the controls
        ctrlbox = wx.BoxSizer(wx.VERTICAL)
        box1 = wx.BoxSizer(wx.HORIZONTAL)
        box2 = wx.BoxSizer(wx.HORIZONTAL)
        # box1 contains the timeslider
        box1.Add(self.timeslider, 1)
        # box2 contains some buttons and the volume controls
        box2.Add(play, flag=wx.RIGHT, border=5)
        box2.Add(pause)
        box2.Add(stop)
        box2.Add(rev)
        box2.Add(ff)
        box2.Add((-1, -1), 1)
        box2.Add(volume)
        box2.Add(self.volslider, flag=wx.TOP | wx.LEFT, border=5)
        # Merge box1 and box2 to the ctrlsizer
        ctrlbox.Add(box1, flag=wx.EXPAND | wx.BOTTOM, border=10)
        ctrlbox.Add(box2, 1, wx.EXPAND)
        ctrlpanel.SetSizer(ctrlbox)
        # Put everything togheter
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.videopanel, 1, flag=wx.EXPAND)
        sizer.Add(ctrlpanel, flag=wx.EXPAND | wx.BOTTOM | wx.TOP, border=10)
        self.SetSizer(sizer)
        self.SetMinSize((10, 10))

        # finally create the timer, which updates the timeslider
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)

        # VLC player controls
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()

    def OnExit(self, evt):
        """
        Closes the window.
        """
        self.Close()
        # end CamStudio recording
        # s.stop()

    def OnOpen(self, evt):
        """
        Pop up a new dialow window to choose a file, then play the selected file.
        """
        global pages
        # if a file is already running, then stop it.
        self.OnStop(None)
        # if there is already a logger, close it

        # Create a file dialog opened in the current home directory, where
        # you can display all kind of files, having as title "Choose a file".
        # dlg = wx.FileDialog(self, "Choose a file", user.home, "",
        #                    "*.*", wx.OPEN)
        dlg = wx.FileDialog(self, "Choose a file", self.basedir, "",
                            "*.mp4", wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            dirname = dlg.GetDirectory()
            self.basedir = dirname
            os.chdir(self.basedir)
            filename = dlg.GetFilename()

            self.basename, junk = os.path.splitext(filename)

            # create logger
            vplogfilename = self.basename + "_VP.log"
            self.logger = logging.getLogger(self.basename)
            hdlr = logging.FileHandler(vplogfilename)
            formatter = logging.Formatter(
                '%(asctime)s %(levelname)s %(message)s')
            hdlr.setFormatter(formatter)
            self.logger.addHandler(hdlr)
            self.logger.setLevel(logging.DEBUG)

            # GET time of each page, for FF/REV
            # pagefilename = self.basename + "pages.txt"
            # if not os.access(pagefilename, os.R_OK):
            #     print('Error: %s file is not present' % pagefilename)
            #     f = open(pagefilename, "w")
            #     call = ["gawk", "-f", "getPages.awk", self.basename + ".log"]
            #     self.logger.info(call)
            #     res = subprocess.call(call, shell=False, stdout=f)
            #     f.close()
            #     if res != 0:
            #         print("Error calling getPages.awk!")
            #         self.logger.debug("Error calling getPages.awk!")
            #         sys.exit(1)
            # pages = np.loadtxt(pagefilename)

            # start CamStudio
            s.setFileName(
                os.path.join(self.basedir, self.basename + "_VP.mp4"))
            s.start()

            # @ Gary Feng
            # rename the SSA file so that the cross won't show
            os.rename(self.basename+".ssa", "X"+self.basename+".ssa")

            # Creation
            self.media = self.instance.media_new(('file:///' +
                                                  unicode(os.path.join(dirname,
                                                                      filename))),
                                                 '--subsdelay-mode=0',
                                                 '--subsdelay-factor={0}'.format(videoDelay))
            self.player.set_media(self.media)

            # garyfeng: set_rate(0.5)
            #self.player.set_rate(0.5)

            # Report the title of the file chosen
            title = self.player.get_title()
            #  if an error was encountred while retriving the title, then use
            #  filename
            if title == -1:
                title = filename
            self.SetTitle("%s - wxVLCplayer" % title)

            # set the window id where to render VLC's video output
            # (use hwnd for windows and xwindow for linux)
            self.player.set_hwnd(self.videopanel.GetHandle())
            self.OnPlay(None)

            # set the volume slider to the current volume
            self.volslider.SetValue(self.player.audio_get_volume() / 2)

        # finally destroy the dialog
        dlg.Destroy()

    def OnPlay(self, evt):
        """
        Toggle the status to Play/Pause.

        If no file is loaded, open the dialog window.
        """
        # check if there is a file to play, otherwise open a
        # wx.FileDialog to select a file
        if not self.player.get_media():
            self.OnOpen(None)
        else:
            # Try to launch the media, if this fails display an error message
            # gary feng: do full screen
            self.player.set_fullscreen(False)
            if self.player.play() == -1:
                self.errorDialog("Unable to play.")
            else:
                if not self.timerInUse:
                    self.timer.Start()
                    self.timerInUse = True
            # now switch to the control panel
            # w.find_window_wildcard(".*"+self.basename+".*")
            # w.set_foreground()

    def OnPause(self, evt):
        """
        Pause the player.
        """
        self.player.pause()

    def OnStop(self, evt):
        """
        Stop the player.
        """
        self.player.stop()
        # reset the time slider
        self.timeslider.SetValue(0)
        self.timer.Stop()
        # gary feng: no full screen
        self.player.set_fullscreen(False)

    def OnNextPage(self, evt):
        """Jump to the next time mark/page"""
        player = self.player
        curTime = player.get_time()
        for p in pages:
            if curTime <= int(p[1]):
                curTime = int(p[1])
                break
        player.set_time(curTime)
        self.setWindowTitle()

    def OnLastPage(self, evt):
        """Jump to the last time mark/page"""
        player = self.player
        curTime = player.get_time()
        lastTime = int(99999)
        for p in pages:
            if curTime <= int(p[1]):
                break
            lastTime = int(p[1])
        player.set_time(lastTime)
        self.setWindowTitle()

    def OnAdjust(self, evt):
        """Change movie time position according to the slider"""
        self.bSliderDragged = True
        val = float(self.timeslider.GetValue()) / (self.timeslider.GetMax() - self.timeslider.GetMin())
        self.player.set_position(float(val))
        # print(val)
        self.bSliderDragged = False
        self.setWindowTitle()

    def OnTimer(self, evt):
        """
        Update the time slider according to the current movie time.
        """
        # disable when slider is being dragged
        if self.bSliderDragged:
            return

        # since the self.player.get_length can change while playing,
        # re-set the timeslider to the correct range.
        length = self.player.get_length()
        self.timeslider.SetRange(-1, length)

        # update the time on the slider
        time = self.player.get_time()
        self.timeslider.SetValue(time)
        # i=self.timer.GetInterval()
        # if i==0: i=1000
        # find out which page you are on
        self.setWindowTitle()

    def setWindowTitle(self):
        page = 0
        player = self.player
        time = player.get_time()
        for p in pages:
            if time <= int(p[1]):
                page = int(p[0])
                break
        self.SetTitle("%s - %i min %i sec @Page %i" % (self.basename, time / 60000, (
            time - int(time / 60000) * 60000) / 1000, page))

    def OnToggleVolume(self, evt):
        """
        Mute/Unmute according to the audio button.
        """
        is_mute = self.player.audio_get_mute()

        self.player.audio_set_mute(not is_mute)
        # update the volume slider;
        # since vlc volume range is in [0, 200],
        # and our volume slider has range [0, 100], just divide by 2.
        self.volslider.SetValue(self.player.audio_get_volume() / 2)

    def OnSetVolume(self, evt):
        """
        Set the volume according to the volume sider.
        """
        volume = self.volslider.GetValue() * 2
        # vlc.MediaPlayer.audio_set_volume returns 0 if success, -1 otherwise
        if self.player.audio_set_volume(volume) == -1:
            self.errorDialog("Failed to set volume")

    def errorDialog(self, errormessage):
        """
        Display a simple error dialog.
        """
        edialog = wx.MessageDialog(self, errormessage, 'Error', wx.OK |
                                   wx.ICON_ERROR)
        edialog.ShowModal()

if __name__ == "__main__":
    curTime = 0
    videoDelay = 1.0
    echo_position = False
    pages = {}


    # Create a wx.App(), which handles the windowing system event loop
    app = wx.PySimpleApp()
    # Create the window containing our small media player
    player = Player("Simple PyVLC Player")
    # set the path to the current script
    player.basedir = os.path.realpath(os.path.dirname(sys.argv[0]))

    # setup ScreenCaster
    s = ScreenCaster("v2.mp4", vlc_instance=player.instance)
    s.openLogger("v2-vlc.log")

    # show the player window centred and run the application
    player.Centre()
    player.Show()
    app.MainLoop()

    # when it's done from MainLoop(), prepare to quit
    s.stop()

