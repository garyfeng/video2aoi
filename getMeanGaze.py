# -*- coding: utf-8 -*-

'''
Extract gaze positions from HDF5 file for use with v2.py or create_subtitle.py
@author: Dan Blanchard
@contact: dblanchard@ets.org
@date: April 2013

Gary Feng, Nov 2013:
- modified to print the average, because iohub switched from top-left to center-based 
  coordinate system at some point. 
'''

from __future__ import print_function, unicode_literals, with_statement

#import numpy
import argparse
import os
from itertools import izip, groupby
from operator import itemgetter

import tables

def hdf5_to_eye_log(hdf5_filename, eye='left', subject=None, experiment=None,
                    output_directory='.'):
    '''
    Takes an HDF5 file and creates eye logs for each subject and experiment in
    file or filters on subject and/or experiment if specified.
    '''
    # Read in tables
    try:
      h5file = tables.openFile(hdf5_filename)
    except:
      print ("Error, {} cannot be opened".format(hdf5_filename))
      return False

    # Build handy dictionaries
    code_subject_dict = dict([(row['code'], row['session_id']) for row in
                               h5file.root.data_collection.session_meta_data])
    subject_code_dict = dict([(row['session_id'], row['code']) for row in
                               h5file.root.data_collection.session_meta_data])
    code_experiment_dict = dict([(row['code'], row['experiment_id']) for row in
                               h5file.root.data_collection.experiment_meta_data])
    # gary feng: startTimes are init-ed to 0
    startTime = dict([(row['session_id'], 0) for row in
                               h5file.root.data_collection.session_meta_data])

    # Build filter string
    filter_string = '(experiment_id > 0)'
    if subject is not None:
        filter_string += ' & (session_id == {0})'.format(code_subject_dict[subject])
    if experiment is not None:
        filter_string += ' & (experiment_id == {0})'.format(code_experiment_dict[experiment])

    # get the eyegaze table first
    table = h5file.root.data_collection.events.eyetracker.BinocularEyeSampleEvent
    # Do query, valid gaze only
    valid_rows = sorted(table.readWhere(filter_string+' & (status==0)'), key=itemgetter('experiment_id',
                                                                       'session_id',
                                                                       'time'))
    # Proceed if we got anything
    if valid_rows:
        # Print all valid gaze data for given eye
        for session_id, group_iter in groupby(valid_rows, key=itemgetter('session_id')):
            # now print the mean x and y locations
            data = [(row['{0}_gaze_x'.format(eye)], row['{0}_gaze_y'.format(eye)]) for row in group_iter]
            x = [gx for (gx, gy) in data]
            y = [gy for (gx, gy) in data]
            print("{} :: {}: n={:d}, x={:d}, y={:d}".format(hdf5_filename, 
              subject_code_dict[session_id], 
              int(len(x)), int(sum(x)/len(x)), int(sum(y)/len(y))))

    # close HDF5 file
    h5file.close()

def main():
    ''' Main function that processes arguments and gets things started. '''
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Extract gaze positions from HDF5 files and print the average x and y positions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')
    parser.add_argument('hdf5_file',
                        help='The HDF5 data file to extract gaze positions ' +
                             'from.', nargs='+')
    parser.add_argument('-e', '--eye',
                        help='The eye whose position we want gazes for.',
                        choices=['left', 'right'],
                        default='left')
    parser.add_argument('-s', '--subject',
                        help='The subject whose eye tracking data you want to \
                              retrive. By default, dumps entire HDF5 file.')
    parser.add_argument('-x', '--experiment',
                        help='The experiment you want to gaze data from. By \
                              default, dumps entire HDF5 file.')
    args = parser.parse_args()

    # Iterate through given files
    for hdf5_filename in args.hdf5_file:
        hdf5_to_eye_log(hdf5_filename, eye=args.eye, experiment=args.experiment,
                        subject=args.subject)

if __name__ == '__main__':
    main()
