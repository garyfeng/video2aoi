# -*- coding: utf-8 -*-

'''
Convert eye tracker log files into SSA/ASS subtitle files.
@author: Dan Blanchard
@contact: dblanchard@ets.org
@date: March 2013
'''

from __future__ import print_function, unicode_literals

import argparse
import sys
from datetime import timedelta
from textwrap import dedent

from win32api import GetSystemMetrics

def ssa_for_eye_log(eye_log, width=GetSystemMetrics(0),
                    height=GetSystemMetrics(1), eye_num=0,
                    output_filename='-', sync=0):
    '''
    Creates a file containing the SSA-subtitle-version of the eye tracking data
    in eye_log.
    '''
    with open(output_filename, 'w') as output_file:
        # Print config section
        preamble = dedent('''
        [Script Info]
        Title: Eye Gaze Positions
        ScriptType: v4.00+
        WrapStyle: 0
        PlayResX: {0}
        PlayResY: {1}
        ScaledBorderAndShadow: yes

        [V4+ Styles]
        Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
        Style: Default,Arial,72,&H000000FF,&H0000FF00,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,10,1

        [Events]
        Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text''').format(width, height)
        print(preamble, file=(output_file if output_filename != '-' else sys.stdout))

        # Print eye gaze data
        prev_time = timedelta(0)
        # gary feng: added parameter to allow for correcting video-gaze asynchrony
        time_offset = timedelta(0) + sync
        first = True
        for line in eye_log:
            line = line.decode('utf-8').strip()
            x_coord, y_coord, time_sec = [float(x) for x in line.split()]
            timestamp = timedelta(0, time_sec)

            # Handle weird thing where ioHub doesn't use zero as start time.
            if first:
                time_offset = timestamp
            timestamp -= time_offset

            # Skip bad lines and first line
            if x_coord > 0 and not first:
                 print(('Dialogue:{0},{1},{2},Default,,0000,0000,0000,' +
                        ',{{\\pos({3},{4})\\an5}}+').format(eye_num,
                                                            unicode(prev_time)[:-4],
                                                            unicode(timestamp)[:-4],
                                                            int(round(x_coord)),
                                                            int(round(y_coord))),
                        file=(output_file if output_filename != '-' else sys.stdout))
            prev_time = timestamp
            first = False


def main():
    ''' Main function that processes arguments and gets things started. '''
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Convert eye tracker log files into SSA/ASS subtitle files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('eye_log',
                        help='Tab-delimited file containing rows of timestamps' +
                              'in milliseconds, X-coordinates, and ' +
                              'Y-coordinates.',
                        type=argparse.FileType('r'), nargs='+')
    parser.add_argument('-h', '--height',
                        help="Height of the screen log is for.", type=int,
                        default=GetSystemMetrics(1))
    parser.add_argument('-w', '--width',
                        help="Width of the screen log is for.", type=int,
                        default=GetSystemMetrics(0))
    parser.add_argument('-s', '--sync',
                        help="Offset to sync video with gaze data", type=int,
                        default=0)
    args = parser.parse_args()

    for eye_num, eye_log in enumerate(args.eye_log):
        ssa_for_eye_log(eye_log, width=args.width, height=args.height,
                        eye_num=eye_num, sync = args.sync)


if __name__ == '__main__':
    main()
