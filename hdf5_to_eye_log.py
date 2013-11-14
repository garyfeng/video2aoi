# -*- coding: utf-8 -*-

'''
Extract gaze positions from HDF5 file for use with v2.py or create_subtitle.py
@author: Dan Blanchard
@contact: dblanchard@ets.org
@date: April 2013

Gary Feng, Nov 2013:
- exporting keystroke and mouse data
- using regex for subject and hdf5 selection
'''

from __future__ import print_function, unicode_literals, with_statement

import argparse
import os
import glob 
from itertools import groupby
from operator import itemgetter
import re
import numpy as np

import tables

def readDataTable(table, subject, filter_string, subject_code_dict):
  '''Read rows of data from the *table* and filter with *subject* and *experiment*.
  :param table: the HDF5 data table 
  :param subject: the regex string to filter by subject name (aka 'code' in iohub)
  :param filter_string: used to select the data in addition to subject
  :param subject_code_dict: a dict built from the HDF5 tables to convert session_id to code
  :Returns: numpy 1-D array -- data read; or None if no subject or data 
  '''

  # Do query, all gaze, including missing or invalid gaze
  # this is necessary if later we want to know the % of missing data
  valid_rows=None
  # need to filter according to subject
  if subject is not None:
    # get the matching subjs
    matchedSubjects = [s for s in subject_code_dict if re.match( subject, subject_code_dict[s], re.I)]
    if len(matchedSubjects)<1:
      # no matching subjects; quit this file
      return None
    for session_id in matchedSubjects:
      # this is an ugly hack to filter by subject_code but it's quick
      # skip the subject if subject pattern is specified but doesn't match this one
      # assume it's a regex pattern
      #matchObj = re.match( subject, subject_code_dict[session_id], re.I)
      #if not matchObj: 
        # the subjname (code) matches, now read
      tmp = table.readWhere(filter_string+' & (session_id == {0})'.format(session_id))
      #print ("HDF={} Subject={} code={} dimensions={}".format(hdf5_filename, session_id, 
      #                      subject_code_dict[session_id], str(tmp.shape)))

      if valid_rows is None:
        valid_rows = tmp
      else:
        #valid_rows = np.column_stack((valid_rows, tmp ))
        valid_rows=np.append(valid_rows,tmp) 
    # now we have all the subjects, we sort
    valid_rows = sorted(valid_rows, key=itemgetter('experiment_id',
                                                                     'session_id',
                                                                     'time'))
  else:
    # if subject is not specified
    valid_rows = sorted(table.readWhere(filter_string), key=itemgetter('experiment_id',
                                                                     'session_id',
                                                                     'time'))
  # return the data table
  return valid_rows if len(valid_rows)>0 else None

def hdf5_to_eye_log(hdf5_filename, eye='left', subject=None, experiment=None,
                    output_directory='.', origin='center', monitor_size='1920x1280'):
    '''
    Takes an HDF5 file and creates eye logs for each subject and experiment in
    file or filters on subject and/or experiment if specified.
    '''
    # Read in tables
    try:
      h5file = tables.openFile(hdf5_filename)
    except:
      print ("Error: {} cannot be opened".format(hdf5_filename))
      return False
    # Parse monitor size
    try:
      monitorW, monitorH=[int(x) for x in monitor_size.split('x')]
    except:
      print("Error: monitor_size paramter '{}' cannot be parsed.".format(monitor_size))
      return False
    
    print ("\nProcessing hdf5 file {}".format(hdf5_filename))

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
    if experiment is not None:
        filter_string += ' & (experiment_id == {0})'.format(code_experiment_dict[experiment])

    ###############################
    # get the eyegaze table first
    ###############################
    table = h5file.root.data_collection.events.eyetracker.BinocularEyeSampleEvent
    # Do query, all gaze, including missing or invalid gaze
    # this is necessary if later we want to know the % of missing data
    valid_rows=readDataTable(table, subject, filter_string, subject_code_dict)

    # Proceed if we got anything
    if valid_rows:
        # Print all valid gaze data for given eye
        for session_id, group_iter in groupby(valid_rows, key=itemgetter('session_id')):

            with open(os.path.join(output_directory, '{0}_eye.txt'.format(subject_code_dict[session_id])),
                      'w') as subject_log:
                gotFirstGaze = False
                c=0
                for row in group_iter:

                    # skip all junk gaze data until the first valid sample
                    if not gotFirstGaze and row['status'] != 0:
                      # skip
                      continue
                    elif not gotFirstGaze and row['status'] == 0:
                      # first gaze reading
                      startTime[session_id] = int(round(row['time']*1000))
                      gotFirstGaze=True 
                    
                    c+=1
                    # not first gaze; subtract the known startTime
                    st=startTime[session_id] 
                    # export missing values if the status !=0
                    if row['status']==0:
                      gazex = int(round(row['{0}_gaze_x'.format(eye)]))
                      gazey = int(round(row['{0}_gaze_y'.format(eye)]))
                      # adjust for origin; y is flipped
                      gazex = gazex + int(monitorW/2) if origin=='center' else gazex
                      gazey = - gazey + int(monitorH/2) if origin=='center' else gazey
                    else:
                      gazex = -32768; gazey=-32768
                    # now export
                    print('{0:d}\t{1}\t{2:d}\t{3:d}\t{4}'.format( int(round(row['time']*1000)-st),
                                                "gaze",
                                                gazex,
                                                gazey,
                                                row["event_id"]),
                                          file=subject_log)
            print ('Gaze: {} gaze events saved to {}_eye.txt, startTime = {}'.format(c, subject_code_dict[session_id], startTime[session_id]))
            subject_log.close()

    ###############################
    # get the keystroke table 
    ###############################
    table = h5file.root.data_collection.events.keyboard.KeyboardCharEvent
    # Do query, using the same filter
    valid_rows=readDataTable(table, subject, filter_string, subject_code_dict)
    # Proceed if we got anything
    if valid_rows:
        # Print all valid gaze data for given eye
        for session_id, group_iter in groupby(valid_rows,
                                           key=itemgetter('session_id')):
            with open(os.path.join(output_directory, '{0}_eye.txt'.format(subject_code_dict[session_id])),
                      'a') as subject_log:
              try:
                c=0
                st=startTime[session_id] 
                for row in group_iter:
                    c+=1
                    if round(row['time']*1000)>st:
                        print('{0:d}\t{1}\t{2}\t{3}\t{4}'.format( int(round(row['time']*1000)-st),
                                                  "keyboard",
                                                  "",
                                                  "",
                                                  row['key']),
                          file=subject_log)
                print ('Keyboard: {} key events saved to {}_eye.txt '.format(c, subject_code_dict[session_id]))
              except:
                print ('Error: keyboard events cannot be saved to {0}_eye.txt '.format(subject_code_dict[session_id]))
            subject_log.close()

    ###############################
    # get the mouse table 
    ###############################
    table = h5file.root.data_collection.events.mouse.MouseInputEvent
    # Do query
    valid_rows=readDataTable(table, subject, filter_string, subject_code_dict)

    # Proceed if we got anything
    if valid_rows:
        # Print all valid gaze data for given eye
        for session_id, group_iter in groupby(valid_rows,
                                           key=itemgetter('session_id')):
            with open(os.path.join(output_directory,
                                   '{0}_eye.txt'.format(subject_code_dict[session_id])),
                      'a') as subject_log:
              try:
                c=0
                for row in group_iter:
                    c+=1
                    st=startTime[session_id] 
                    mousex = int(round(row["x_position"]))
                    mousey = int(round(row["y_position"]))
                    # adjust for origin 
                    mousex = mousex + int(monitorW/2) if origin=='center' else mousex
                    mousey = - mousey + int(monitorH/2) if origin=='center' else mousey
                    if round(row['time']*1000)>st:
                        print('{0:d}\t{1}\t{2}\t{3}\t{4}'.format( int(round(row['time']*1000)-st),
                                                  "mouse",
                                                  mousex,
                                                  mousey,
                                                  row['pressed_buttons']),
                          file=subject_log)
                print ('Mouse: {} mouse events saved to {}_eye.txt '.format(c, subject_code_dict[session_id]))
              except:
                print ('Error: mouse events cannot be saved to {0}_eye.txt '.format(subject_code_dict[session_id]))
            subject_log.close()

    ###############################
    # close HDF5 file
    ###############################
    h5file.close()

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
                             'from. This can be a glob pattern, e.g., "event*.hdf5".' +
                             'And you can do "study1*.hdf5+expt2*.hdf5".', nargs='+')
    parser.add_argument('-e', '--eye',
                        help='The eye whose position we want gazes for.',
                        choices=['left', 'right'],
                        default='left')
    parser.add_argument('-s', '--subject',
                        help='A regular expression of the subjects whose eye tracking data you want to \
                              retrive. By default, dumps entire HDF5 file.')
    parser.add_argument('-x', '--experiment',
                        help='The experiment you want to gaze data from. By \
                              default, dumps entire HDF5 file.')
    parser.add_argument('-o', '--origin', 
                        choices=['topleft', 'center'],
                        default='center',
                        help='The origin of the screen coordinate system. By \
                              default, it is the center (per the current iohub setting).\
                              If "topleft", then use the top-left corner.')
    parser.add_argument('-m', '--monitor_size',
                        default = '1920x1080',
                        help='The resolution of the primary monitor. By \
                              default, it is 1920x1080).\
                              Specify in the format "WxH".')
    args = parser.parse_args()

    # Iterate through given files
    for hdf5_filename in args.hdf5_file:
      for h in glob.glob(hdf5_filename):
        hdf5_to_eye_log(h, eye=args.eye, experiment=args.experiment,
                        subject=args.subject, 
                        origin = args.origin, 
                        monitor_size=args.monitor_size)

if __name__ == '__main__':
    main()
