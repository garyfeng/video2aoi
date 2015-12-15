'''
Module for recording screen content with libvlc.

@author: Dan Blanchard
@author: Gary Feng
'''

from __future__ import print_function, unicode_literals

import glob
import logging
import os
import os.path
import subprocess
import sys
import time

import vlc


class ScreenCaster(object):
    """
    A class for recording the screen using LibVLC and the
    screen-capture-recorder DirectShow filter.
    """

    def __init__(self, output_filename, fps=40, encoder='h264',
                 audio_device='microphone',
                 video_device='screen-capture-recorder', live_caching=300,
                 vlc_instance=None):
        '''
        Initialize the VLC instance we're using to capture

        @param fps: The framerate of the video.
        @param encoder: The video encoder to Use
        @param audio_device: The audio capture device to use.
        @param video_device: The video capture device to use.
        @param live_caching: The number of milliseconds to use for caching the
                             video and audio.
        @param output_filename: The filename to save the video to.
        '''
        self._fps = fps
        self._encoder = encoder
        self._audio_device = audio_device
        self._video_device = video_device
        self._live_caching = live_caching
        self._output_filename = output_filename
        self._vlc_instance = vlc.Instance() if vlc_instance is None else vlc_instance
        self._vlc_player = None
        self._recording = False
        self._aspect_ratio = "4\\:3"  # TODO: Figure this out on the fly

    def start(self):
        # Start recording if asked
        if not self._recording:
            try:
                media = self._vlc_instance.media_new(
                    "dshow://", ":dshow-vdev={0}".format(self._video_device),
                    ":dshow-aspect-ratio={0}".format(self._aspect_ratio),
                    ":live-caching={0}".format(self._live_caching),
                    # ":dshow-fps={0}".format(self._fps),
                    ":sout-x264-preset=stillimage",
                    (":sout=#transcode{{vcodec={0},vb=0,scale=0,acodec=mp4a," +
                        "ab=128,channels=2,samplerate=44100,fps={1}}}:std{{" +
                        "access=file,mux=mp4,dst='{2}'}}").format(self._encoder,
                            self._fps, self._output_filename))
                if self._audio_device:
                    media.add_option(":dshow-adev=".format(self._audio_device))
                self._vlc_player = self._vlc_instance.media_player_new()
                self._vlc_player.set_media(media)
                if self._vlc_player.play() == -1:
                    raise Exception()
                self._recording = True
                # log the time
                logging.info("info\tScreenCaster.start")
            except Exception, e:
                logging.error("error\tScreenCaster.start: Unable to start recording")
                print("Error starting VLC screen capture.", file=sys.stderr)
                raise e


    def stop(self):
        if self._recording:
            self._vlc_player.stop()
            self._recording = False

            # log the time
            logging.info("info\tScreenCaster.stop: ")

    def wait(self, t):
        time.sleep(t)

    def openLogger(self, logname=""):

        # configure default root logger to log all debug messages to stdout
        # gary feng
        loglevel = logging.INFO
        logformatter = logging.Formatter("%(asctime)s\t%(message)s")
        # logging.basicConfig(filename=logname,level=loglevel)
        rootlogger = logging.getLogger('')
        # rootlogger.setLevel(loglevel)

        # consolehandler = logging.StreamHandler(sys.stdout)
        # consolehandler.setLevel(loglevel)
        # consolehandler.setFormatter(logformatter)
        # rootlogger.addHandler(consolehandler)

        if (logname != ""):
            systemloghandler = logging.FileHandler(logname)
            systemloghandler.setLevel(loglevel)
            systemloghandler.setFormatter(logformatter)
            rootlogger.addHandler(systemloghandler)

            # print("ScreenCaster:OpenLogger"+logname)

        if(rootlogger is not None):
            self.logger = rootlogger
        else:
            print("Can't find root logger", file=sys.stderr)
            raise Exception()

    def setFileName(self, filename):
        self._output_filename = filename


if __name__ == '__main__':
    s = ScreenCaster("test.mp4")
    s.openLogger("test.log")
    s.start()
    s.wait(4)
    s.stop()