"""
Modified version of Headless example from ioHub that uses our ScreenCaster.py device for screen recording.
"""
import errno
import os
import time

import ioHub
from ioHub.devices import Computer
from ioHub.constants import EventConstants
from ioHub.util.experiment import ioHubExperimentRuntime
from ScreenCaster import ScreenCaster
from hdf5_to_eye_log import hdf5_to_eye_log
from create_subtitle import ssa_for_eye_log


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError, exception:
        if exception.errno != errno.EEXIST:
            raise exception


class ExperimentRuntime(ioHubExperimentRuntime):
    """
    """
    def run(self,*args,**kwargs):
        """
        Start recording
        """
        # Setup devices
        self.keyboard=self.devices.kb
        self.display=self.devices.display
        self.mouse=self.devices.mouse
        self.eyetracker=self.devices.tracker
        self.print_eye_event_stream=False
        self.print_current_gaze_pos=False
        self.print_keyboard_event_stream=False
        self.print_mouse_event_stream=False
        self.directory = self.configuration['code']
        # Create the directory for storing results if needed
        make_sure_path_exists(self.directory)

        # Setup screen caster
        self.basename = self.configuration['session_defaults']['code']
        self.screen_caster = ScreenCaster(os.path.join(self.directory,
                                                       '{0}.mp4'.format(self.basename)))
        self.screen_caster.openLogger()
        self.app_start_time = Computer.getTime()


        # Loop until we get a keyboard event with the space, Enter (Return),
        # or Escape key is pressed.

        self.printCommandOptions()
        self.printApplicationStatus()

        if not self.runEyeTrackerCalibration():
            # Wait for screen to come back after calibration
            time.sleep(1)

            # Start recording
            self.toggleScreenRecording()
            time.sleep(1.5)  # Wait for screen recorder to start
            self.toggleEyeEventRecording()

            # Main event loop
            while 1:

                 # if 'an event handler returns True, quit the program
                if self.handleEyeTrackerEvents():
                    break
                if self.handleKeyboardEvents():
                    break
                if self.handleMouseEvents():
                    break

                if self.eyetracker:
                    self.printGazePosition()

                # discard any event we do not need from the online event queues
                self.hub.clearEvents('all')

                # since realtime access to events on this process does not matter in
                # this demo, elt's sleep for 20 msec every loop
                time.sleep(0.02)

        # Stop recording
        self.toggleEyeEventRecording()
        self.toggleScreenRecording()
        self.eyetracker.setConnectionState(False)
        self.hub.clearEvents('all')

        # Update HDF5 file
        self.hub.flushIODataStoreFile()

        # After everything's done using events.hdf5, create subtitles for the
        # current subject
        hdf5_to_eye_log('events.hdf5', subject=self.basename,
                        experiment=self.configuration['code'],
                        output_directory=self.directory)

        # Create subtitle file
        with open(os.path.join(self.directory,
                               self.basename + '_eye.log')) as eye_log:
            ssa_for_eye_log(eye_log,
                            output_filename=os.path.join(self.directory,
                                                         self.basename + '.ssa'),
                            width=1280, height=1024)

        # END OF PROGRAM

    def handleKeyboardEvents(self):
        event_list=self.keyboard.getEvents(EventConstants.KEYBOARD_CHAR)

        for k in event_list:
            if k.modifiers is not None and 'CONTROL_LEFT' in k.modifiers:
                if k.key in ['Q']:
                    print '\n>> Quit key pressed:', k.key, 'for', k.duration, 'sec.'
                    return True
                elif k.key == 'E':
                    self.toggleEyeEventPrinting()
                elif k.key == 'G':
                    self.toggleGazePositionPrinting()
                elif k.key == 'R':
                    return self.runEyeTrackerCalibration()
                elif k.key == 'H':
                    self.printCommandOptions()
                elif k.key == 'S':
                    self.printApplicationStatus()
            elif self.print_keyboard_event_stream is True:
                self.printKeyEvent(k)

        return False

    def handleMouseEvents(self):
        if self.print_mouse_event_stream is True:
            event_list=self.mouse.getEvents()
            for m in event_list:
                self.printMouseEvent(m)

        return False

    def handleEyeTrackerEvents(self):
        event_list=self.eyetracker.getEvents()
        if self.print_eye_event_stream is True:
            for t in event_list:
                self.printEyeEvent(t)
        return False

    def toggleGazePositionPrinting(self):
        print ''
        self.print_current_gaze_pos=not self.print_current_gaze_pos
        if self.print_current_gaze_pos is True:
            print '\n>> Disabled ALL Event Printing.'
            self.print_eye_event_stream=False
            self.print_mouse_event_stream=False
            self.print_keyboard_event_stream=False
        print '>> Gaze Pos Print Status: ',self.print_current_gaze_pos
        print '>> Eye Event Print Status: ',self.print_eye_event_stream
        print ''

    def printEyeEvent(self,e):
        if e.type == EventConstants.BINOCULAR_EYE_SAMPLE:
            print 'BINOC SAMPLE:\n\tLEFT:\n\t\tSTATUS: %d\n\t\tPOS: (%.3f,%.3f)\n\t\tPUPIL_SIZE: %.3f\n\tRIGHT:\n\t\tSTATUS: %d\n\t\tPOS: (%.3f,%.3f)\n\t\t PUPIL_SIZE: %.3f\n'%(
                     int(e.status / 10.0),
                     e.left_gaze_x,e.left_gaze_y,
                     e.left_pupil_measure1,
                     e.status%10,
                     e.right_gaze_x,e.right_gaze_y,
                     e.right_pupil_measure1)
        elif e.type == EventConstants.MONOCULAR_EYE_SAMPLE:
            print 'MONOC SAMPLE: mot handled yet'
        else:
            print 'Unhandled eye event:\n{0}\n'.format(e)

    def printGazePosition(self):
        if self.print_current_gaze_pos is True and self.eyetracker.isRecordingEnabled():

            gp=self.eyetracker.getLastGazePosition()
            if gp is None:
                print 'GAZE POS: TRACK LOSS\r',
            else:
                gx,gy=gp
                print 'GAZE POS: ( %.3f, %.3f )\r'%(gx,gy),

    def toggleEyeEventPrinting(self):
        print ''
        self.print_eye_event_stream=not self.print_eye_event_stream
        if self.print_eye_event_stream is True:
            print '\n>> Disabled Gaze Pos Printing.'
            self.print_current_gaze_pos=False
        print '>> Eye Event Print Status: ',self.print_eye_event_stream
        print '>> Gaze Pos Print Status: ',self.print_current_gaze_pos
        print ''

    def toggleMouseEventPrinting(self):
        print ''
        self.print_mouse_event_stream=not self.print_mouse_event_stream
        if self.print_current_gaze_pos is True:
            print '\n>> Disabled Gaze Pos Printing.'
            self.print_current_gaze_pos=False
        print '>> Mouse Event Print Status: ',self.print_mouse_event_stream
        print ''

    def printMouseEvent(self,m):
        print 'Mouse Event: ', m.time, (m.x_position,m.y_position), m.button_id, m.wheel_value, m.window_id

    def toggleKeyboardEventPrinting(self):
        print ''
        self.print_keyboard_event_stream=not self.print_keyboard_event_stream
        if self.print_current_gaze_pos is True:
            print '\n>> Disabled Gaze Pos Printing.'
            self.print_current_gaze_pos=False
        print '\n>> Keyboard Event Print Status: ',self.print_keyboard_event_stream
        print ''

    def printKeyEvent(self,k):
        print 'Keyboard Event: ', k.time, k.key, k.modifiers

    def toggleEyeEventRecording(self):
        # Toggle eye tracker recording.
        is_recording=self.eyetracker.setRecordingState(not self.eyetracker.isRecordingEnabled())
        print "\n>> NOTE: Tracker Recording State change: ",is_recording

    def toggleScreenRecording(self):
        # Toggle screen recording.
        if self.screen_caster._recording:
            self.screen_caster.stop()
        else:
            self.screen_caster.start()
        print "\n>> NOTE: Screen Recording State change: ", self.screen_caster._recording

    def runEyeTrackerCalibration(self):
        # start calibration if not recording data
        if self.eyetracker:
            if self.eyetracker.isRecordingEnabled() is True:
                print '\n>> ERROR: Can not calibrate when recording. Stop recording first, then calibrate'
                return False

            calibrationOK=self.eyetracker.runSetupProcedure()
            if calibrationOK is False:
                print "\n>> ERROR: Exiting application due to failed calibration."
                return True

        print "\n>> NOTE: Tracker Calibration Done."

    def printCommandOptions(self):
        print ''
        print '######################################'
        print '# >> Headless ioHub Controls:        #'
        print '#                                    #'
        print '# The following key combinations     #'
        print '# are available:                     #'
        print '#                                    #'
        print '# L_CTRL+Q:      End Program         #'
        print '# L_CTRL+R:      Recalibrate tracker #'
        print '#                                    #'
        print '# L_CTRL+E:      Toggle Eye Events   #'
        print '# L_CTRL+G:      Toggle Gaze Pos.    #'
        print '#                                    #'
        print '# L_CTRL+H:      Print Controls      #'
        print '# L_CTRL+S:      Print Status        #'
        print '#                                    #'
        print '######################################'
        print ''

    def printApplicationStatus(self):
        print ''
        print 'Headless ioHub Status:'
        if self.eyetracker:
            print '\tRunning Time: %.3f seconds.'%(Computer.getTime()-self.app_start_time)
            print '\tRecording Eye Data: ',self.eyetracker.isRecordingEnabled()
            print '\tRecording Screen: ',self.screen_caster._recording
            print '\tPrinting Eye Events: ',self.print_eye_event_stream
            print '\tPrinting Gaze Position: ',self.print_current_gaze_pos
        print ''

    def prePostSessionVariableCallback(self,sessionVarDict):
        sess_code=sessionVarDict['code']
        scount=1
        while self.isSessionCodeNotInUse(sess_code) is True:
            sess_code='%s-%d'%(sessionVarDict['code'],scount)
            scount+=1
        sessionVarDict['code']=sess_code
        return sessionVarDict

################################################################################
# The below code should never need to be changed, unless you want to get command
# line arguements or something.

if __name__ == "__main__":
    def main(configurationDirectory):
        """
        Creates an instance of the ExperimentRuntime class, checks for an experiment config file name parameter passed in via
        command line, and launches the experiment logic.
        """
        import sys
        if len(sys.argv)>1:
            configFile=sys.argv[1]
            runtime=ExperimentRuntime(configurationDirectory, configFile)
        else:
            runtime=ExperimentRuntime(configurationDirectory, "experiment_config.yaml")

        runtime.start()

    configurationDirectory=ioHub.module_directory(main)

    # run the main function, which starts the experiment runtime
    main(configurationDirectory)

    # After everything's done using events.hdf5, create subtitles for the
    # current subject
