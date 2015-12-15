'''
Module for recording screen content with libvlc.

@author: Dan Blanchard
@author: Gary Feng
'''

import ioHub
from ioHub.devices import Device
from ioHub.constants import DeviceConstants
import vlc


class ScreenCasterDevice(Device):
    """
    A device for recording the screen using LibVLC and the
    screen-capture-recorder DirectShow filter.
    """
    DEVICE_TYPE_ID = DeviceConstants.OTHER
    DEVICE_TYPE_STRING = 'SCREEN_CASTER'

    __slots__ = ['_vlc_instance', '_encoder', '_fps', '_audio_device',
                 '_video_device', '_live_caching', '_output_filename',
                 '_recording', '_vlc_player', '_aspect_ratio']
    def __init__(self, *args, **kwargs):
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
        self._fps = kwargs.pop('fps')
        self._encoder = kwargs.pop('encoder')
        self._audio_device = kwargs.pop('audio_device')
        self._video_device = kwargs.pop('video_device')
        self._live_caching = kwargs.pop('live_caching')
        Device.__init__(self,*args,**kwargs)
        self._output_filename = self._configuration['output_file']
        self._vlc_instance = vlc.Instance()
        self._recording = False
        self._aspect_ratio = "4\\:3"  # TODO: Figure this out on the fly

        # stores the eye tracker runtime related configuration settings from
        # the ioHub .yaml config file
        # self._runtime_settings=kwargs['dconfig']['runtime_settings']

    def setRecordingState(self, recording):
        """
        The setRecordingState method is used to start or stop the recording of
        the screen and microphone.

        @param recording: if True, the screen caster will start recordng data.;
                          false = stop recording data and save the movie.
        @type recording: bool

        @return:  the current recording state of the screen capture device
        @rtype: bool
        """

        if not isinstance(recording,bool):
            return ioHub.server.createErrorResult(
                "INVALID_METHOD_ARGUMENT_VALUE",
                error_message="The recording arguement value provided is not " +
                    "a boolean.",
                method="EyeTracker.setRecordingState",
                arguement='recording',
                value=recording)

        # Start recording if asked
        if recording and not self._recording:
            try:
                media = self._vlc_instance.media_new(
                    "dshow://", ":dshow-vdev={0}".format(self._video_device),
                    ":dshow-adev=".format(self._audio_device),
                    # ":dshow-aspect-ratio={0}".format(self._aspect_ratio),
                    ":live-caching={0}".format(self._live_caching),
                    (":sout=#transcode{{vcodec={0},vb=0,scale=0,acodec=mpga," +
                        "ab=128,channels=2,samplerate=44100,fps={1}}}:std{{" +
                        "access=file,mux=mp4,dst='{2}'}}").format(self._encoder,
                            self._fps, self._output_filename))
                self._vlc_player = self._vlc_instance.media_player_new()
                self._vlc_player.set_media(media)
                self._recording = True
            except Exception, e:
                return ioHub.server.createErrorResult("IOHUB_DEVICE_EXCEPTION",
                    error_message="An unhandled exception occurred on the " +
                        "ioHub Server Process.",
                    method="ScreenCaster.setRecordingState", error=e)

        # Stop recording if asked and we are
        elif not recording and self._recording:
            try:
                self._vlc_player.stop()
                self._recording = False
            except Exception, e:
                return ioHub.server.createErrorResult("IOHUB_DEVICE_EXCEPTION",
                    error_message="An unhandled exception occurred on the " +
                        "ioHub Server Process.",
                    method="ScreenCaster.setRecordingState", error=e)

        return self._recording
