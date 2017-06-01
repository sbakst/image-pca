from __future__ import absolute_import, division, print_function

import os
import re
import numpy as np
import glob
from ultratils.exp import Exp
import audiolabel
import argparse
from scipy.spatial import distance
from scipy import ndimage
from sklearn.metrics.pairwise import euclidean_distances
import itertools
import subprocess

# for plotting
from PIL import Image
import matplotlib.pyplot as plt

# for PCA business
from sklearn import decomposition
from sklearn.decomposition import PCA

vre =  re.compile(
         "^(?P<vowel>AA|AE|AH|AO|EH|ER|EY|IH|IY|OW|UH|UW)(?P<stress>\d)?$"
      )


# Read in and parse the arguments, getting directory info and whether or not data should flop

try:
    expdir = '/media/sf_bce15/305_toy_block1/'
except IndexError:
    print("\tDirectory provided doesn't exist!")
    sys.exit(2)

# TODO right now this can only be run on one subject's baseline phase at a time. this should loop over subjects.
# TODO alternately, accept STDIN argument for a single subject or series of subjects
# for s in subjects:
#    ... initialize PCA, do PCA, output

e = Exp(expdir)
e.gather()

frames = None
threshhold = 0.020 # threshhold value in s for moving away from acoustic midpoint measure

# subject = [] TODO add this in once subject loop is added 
#phase = []
trial = []
phone = []
tstamp = []

if args.convert:
    conv_frame = e.acquisitions[0].image_reader.get_frame(0)
    conv_img = test.image_converter.as_bmp(conv_frame)

for idx,a in enumerate(e.acquisitions):

    if frames is None:
        if args.convert:
            frames = np.empty([len(e.acquisitions)] + list(conv_img.shape))
        else:
            frames = np.empty([len(e.acquisitions)] + list(a.image_reader.get_frame(0).shape)) * np.nan
    
    a.gather()


    tg = str(a.abs_image_file + ".ch1.TextGrid")
    pm = audiolabel.LabelManager(from_file=tg, from_type="praat")
    print(pm)
    v,m = pm.tier('phone').search(vre, return_match=True)[-1] # return last V = target V
    print(v)
#    print('here comes m')
#    print(m)
    #print( pm.tier('phone'))
    word = pm.tier('word').label_at(v.center).text
    print(word)
#    print(phone)    
#    phase.append(a.runvars.phase)
    trial.append(idx)
    tstamp.append(a.timestamp)
    phone.append(v.text)
    
    if args.convert:
        mid, mid_lab, mid_repl = a.frame_at(v.center,missing_val="prev", convert=True)
    else:
        mid, mid_lab, mid_repl = a.frame_at(v.center,missing_val="prev")

        
    if mid is None:
        if mid_repl is None:
            print("SKIPPING: No frames to re-select in {:}".format(a.timestamp))
            continue
        else:
            if abs(mid_lab.center - v.center) > threshhold:
                print("SKIPPING: Replacement frame past threshhold in {:}".format(a.timestamp))
                continue
            else:
                mid = mid_repl
                
    frames[idx,:,:] = mid

frames = np.squeeze(frames)
trial = np.squeeze(np.array(trial))
tstamp = np.squeeze(np.array(tstamp))



# remove any indices for all objects generated above where frames have NaN values (due to skipping or otherwise)
keep_indices = np.where(~np.isnan(frames).any(axis=(1,2)))[0]
kept_trial = np.array(trial,str)[keep_indices]
kept_frames = frames[keep_indices]
kept_tstamp = tstamp[keep_indices]

myframes = ndimage.median_filter(kept_frames, 5) # comment out if no denoising median filter desired
#           frames[idx,:,:] = mid

for s in range(0, (len(myframes))):
    (myframes)[s][0:100]=0 # This is the number thing
    
base = []
ramp = []
hold = []
wash = []

base_max = np.where(keep_indices==14)[0]
ramp_max = np.where(keep_indices==44)[0]
hold_max = np.where(keep_indices==74)[0]
base = myframes[0:(base_max+1),]
ramp = myframes[(base_max+1):(ramp_max+1),]
hold = myframes[(ramp_max+1):(hold_max+1),]
altered = myframes[(base_max+1):(hold_max+1),]
wash = myframes[(hold_max+1):(len(myframes)+1),]

print(len(base))
print(len(ramp))
print(len(hold))
print(len(wash))
print(len(altered))

basetuplist=[x for x in itertools.combinations(base, 2)]
difflist=[]
for g,h in basetuplist:
    difflist.append(np.linalg.norm(g-h))
print(np.mean(difflist))   
print(np.std(difflist))
#print(np.min(difflist))
#print(np.max(difflist))
print(np.max(difflist)-np.min(difflist))

ramptuplist=[x for x in itertools.combinations(ramp, 2)]
difflist=[]
for g,h in ramptuplist:
    difflist.append(np.linalg.norm(g-h))
print(np.mean(difflist))   
print(np.std(difflist))
#print(np.min(difflist))
#print(np.max(difflist))
print(np.max(difflist)-np.min(difflist))

holdtuplist=[x for x in itertools.combinations(hold, 2)]
difflist=[]
for g,h in holdtuplist:
    difflist.append(np.linalg.norm(g-h))
print(np.mean(difflist))   
print(np.std(difflist))
#print(np.min(difflist))
#print(np.max(difflist))
print(np.max(difflist)-np.min(difflist))

washtuplist=[x for x in itertools.combinations(wash, 2)]
difflist=[]
for g,h in washtuplist:
    difflist.append(np.linalg.norm(g-h))
print(np.mean(difflist))
print(np.std(difflist))
#print(np.min(difflist))
#print(np.max(difflist))
print(np.max(difflist)-np.min(difflist))

alttuplist=[x for x in itertools.combinations(altered, 2)]
difflist=[]
for g,h in alttuplist:
    difflist.append(np.linalg.norm(g-h))
print(np.mean(difflist))   
print(np.std(difflist))
#print(np.min(difflist))
#print(np.max(difflist))
print(np.max(difflist)-np.min(difflist))

for i in range(0,len(altered)):
    print(np.linalg.norm((altered[i])-(altered[i-1])))
    
ktuplist=[x for x in itertools.combinations(myframes, 2)]
difflist=[]
for g,h in ktuplist:
    difflist.append(np.linalg.norm(g-h))
print(np.mean(difflist))   
print(np.std(difflist))
print(np.min(difflist))
print(np.max(difflist))


f1list=[]
for idx,a in enumerate(e.acquisitions):
    wav = a.abs_image_file + '.ch1.wav'
    tg = str(a.abs_image_file + ".ch1.TextGrid")
    pm = audiolabel.LabelManager(from_file=tg, from_type="praat")
    print(tg)
    v,m = pm.tier('phone').search(vre, return_match=True)[-1] # return last V = target V
    print(v.center)
    fldmap=(
            "t1", "0.4f",
            "rms", "s",
            "f1", "s",
            "f2", "s",
            "f3", "s",
            "f4", "s",
            "f0", "s",
            )
    head = 't'.join(fldmap[0:len(fldmap):2]) + '\n'
    # Format string used for output
    fmt = '\t'.join( \
            [ \
            '{' + '{0}:{1}'.format(col,fmt) + '}' \
        for col, fmt in zip( \
            fldmap[0:len(fldmap):2], \
            fldmap[1:len(fldmap):2] \
                ) \
                ] \
              ) + '\n'
    
    tempifc = '__temp.ifc'
        
    speaker = 'male'
    ifc_args = ['ifcformant',
        '--speaker=' + speaker,
        '-e', 'gain -n -3 sinc -t 10 60 contrast',
        '--print-header',
        '--output=' + tempifc]
    proc = subprocess.Popen(ifc_args + [wav], stderr=subprocess.PIPE)
    proc.wait()
    if proc.returncode != 0:
        for line in proc.stderr:
            sys.stderr.write(line + '\n')
        raise Exception("ifcformant exited wtih status: {0}".format(proc.returncode))
    ifc = audiolabel.LabelManager(from_file=tempifc, from_type='table', t1_col='sec')
    f1 = (ifc.tier('f1').label_at(v.center)).text
    print(f1)
    f1list.append(float(f1))
print(f1list)
    
print(np.min(f1list))
val, idx = min((val,idx) for (idx, val) in enumerate(f1list))
print(val)
print(idx)

print(enumerate(f1list))
print(np.linalg.norm((kept_frames[65])-(np.mean(base, axis = 0))))
print(np.linalg.norm(np.mean(base, axis = 0)))
for i in range (0, len(base)):
    print(np.linalg.norm(base[i]))