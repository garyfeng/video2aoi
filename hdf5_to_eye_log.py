# -*- coding: utf-8 -*-

'''
Extract gaze positions from HDF5 file for use with v2.py or create_subtitle.py
@author: Dan Blanchard
@contact: dblanchard@ets.org
@date: April 2013
'''

from __future__ import print_function, unicode_literals, with_statement

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
    h5file = tables.openFile(hdf5_filename)
    table = h5file.root.data_collection.events.eyetracker.BinocularEyeSampleEvent

    # Build handy dictionaries
    code_subject_dict = dict([(row['code'], row['session_id']) for row in
                               h5file.root.data_collection.session_meta_data])
    subject_code_dict = dict([(row['session_id'], row['code']) for row in
                               h5file.root.data_collection.session_meta_data])
    code_experiment_dict = dict([(row['code'], row['experiment_id']) for row in
                               h5file.root.data_collection.experiment_meta_data])

    # Build filter string
    filter_string = '(status == 0)'
    if subject is not None:
        filter_string += ' & (session_id == {0})'.format(code_subject_dict[subject])
    if experiment is not None:
        filter_string += ' & (experiment_id == {0})'.format(code_experiment_dict[experiment])

    # Do query
    valid_rows = sorted(table.readWhere(filter_string), key=itemgetter('experiment_id',
                                                                       'session_id',
                                                                       'time'))

    # Proceed if we got anything
    if valid_rows:
        # Print all valid gaze data for given eye
        for session_id, group_iter in groupby(valid_rows,
                                           key=itemgetter('session_id')):
            with open(os.path.join(output_directory,
                                   '{0}_eye.log'.format(subject_code_dict[session_id])),
                      'w') as subject_log:
                for row in group_iter:
                    print('{0}\t{1}\t{2}'.format(row['{0}_gaze_x'.format(eye)],
                                                 row['{0}_gaze_y'.format(eye)],
                                                 row['time']),
                          file=subject_log)

def main():
    ''' Main function that processes arguments and gets things started. '''
    # Get command line arguments
    parser = argparse.ArgumentParser(
        description="Extract gaze positions from HDF5 files for use with v2.py \
                     or create_subtitle.py.",
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
