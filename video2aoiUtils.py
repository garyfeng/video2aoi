# video2aoiUtils
import argparse
import logging
import yaml

#global yamlconfig

def getColorPlane(yamlconfig):
    '''Returns the color plane code specified in the YAML file; 
    requres global yamlconfig

    :returns: the color plane: default = -99 = all colors; 0 = Blue; 1=Green, 2=Red; -1 = something wrong

    '''

    logging.getLogger('')
    colorPlane = -99; #use all colors
    if "useGrayscaleImage" in yamlconfig["study"].keys() and yamlconfig["study"]["useGrayscaleImage"]==True:
        colorPlane = -1
    elif "useColorChannel" in yamlconfig["study"].keys():
        if yamlconfig["study"]["useColorChannel"] == "B":
            colorPlane = 0
        elif yamlconfig["study"]["useColorChannel"] == "G":
            colorPlane = 1
        elif yamlconfig["study"]["useColorChannel"] == "R":
            colorPlane = 2
        else:
            colorPlane = -1
    logging.info("ColorPlane = "+str(colorPlane))
    return colorPlane    



if __name__ == "__main__":
    # unit testing

    ''' Main function that processes arguments and gets things started. '''
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Usage: python video2aoi.py -y config.yaml videofile.avi .",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('avifiles',
                        help='The video file(s) to process.', nargs='+')
    parser.add_argument('-l', '--logLevel',
                        help='The level of informatin to be logged.',
                        choices=['INFO', 'DEBUG'],
                        default='INFO')
    parser.add_argument('-f', '--startFrame',
                        help='The frame number to start processing.', default='ignore')
    parser.add_argument('-j', '--jumpAhead',
                        help='The # of seconds to jump ahead in the skimming mode.', default='ignore')
    parser.add_argument('-o', '--offsetTime',
                        help='The msec that the video is behind the gaze timestamp.', default='ignore')
    parser.add_argument('-c', '--colorPlane',
                        help='The color plan to use for matching.',
                        choices=['ALL', 'R', 'G', 'B'],
                        default='ALL')
    parser.add_argument('-v', '--videoPlayback',
                        help='Whether to play the video or process silently.',
                        choices=['T', 'F'],
                        default='T')
    parser.add_argument('-y', '--YAMLFile',
                        default = 'default.yaml',
                        help='Name of the YAML configuration file.')
    parser.add_argument('-t', '--testingMode',
                        help='If true, no output; for testing only.',
                        choices=['T', 'F'],
                        default='F')
    parser.add_argument('-m', '--mouseBasedTimeSync',
                        help='Whether to use the mouse to sync gaze and video data.',
                        choices=['T', 'F'],
                        default='T')
    parser.add_argument('-s', '--outputLogFileSuffix',
                        help='Suffix for the output log file.',
                        default='ignore')
    args = parser.parse_args()

    #################################
    # now process the args
    #################################

    yamlfile = args.YAMLFile
    try:
        yamlconfig = yaml.load(open(yamlfile))
    except:
        print "Error with the YAML file: {} cannot be opened.".format(yamlfile)
        exit(-1)
    assert "tasks" in yamlconfig
    assert "study" in yamlconfig

    print "TestingGetColorPlane() = {}".format(getColorPlane(yamlconfig))
