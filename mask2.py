"""
5-9-2016 Matthew Faytak
This script reads midpoint frames in from an experiment, performs PCA on the image data, 
and returns a .csv file containing PC loadings and associated metadata. Does not currently
run properly if run on multiple subjects at once, since pca object is initialized with all
data in experiment directory. 

10-4-2016 Sarah Bakst
Revisions specifically for FUSP experiment pilot.
pilot

5-2-2017 Sarah Bakst
Applies mask
subject loop
----
Expected usage if stored in processing dir: $ python image_pca.py . (-f -v)
----
The file you should be loading from is /media/sf_python/feedback/*.ipynb

"""
from __future__ import absolute_import, division, print_function

import os
import re
import numpy as np
from ultratils.exp import Exp
import audiolabel
import argparse

# for plotting
from PIL import Image
import matplotlib.pyplot as plt

# for PCA business
from sklearn import decomposition
from sklearn.decomposition import PCA

vre =  re.compile(
         "^(?P<vowel>AA|AE|AH|AO|EH|ER|EY|IH|IY|OW|EH1|UH|UW)(?P<stress>\d)?$"
      )
     
    

# Read in and parse the arguments, getting directory info and whether or not data should flop
parser = argparse.ArgumentParser()
parser.add_argument("directory", help="Experiment directory containing all subjects")
parser.add_argument("-v", "--visualize", help="Produce plots of PC loadings on fan",action="store_true")
parser.add_argument("-f", "--flop", help="Horizontally flip the data", action="store_true")
parser.add_argument("-c", "--convert", help="Scan-convert the data before analysis", action="store_true")
#parser.add_argument("-r", "--no_er", help="Run PCA without schwar in data set.", action="store_true")
parser.add_argument("outfile", help="Desired output file, intended as .csv")
args = parser.parse_args()

try:
    directory = args.directory
except IndexError:
    print("\tDirectory provided doesn't exist!")
    ArgumentParser.print_usage
    ArgumentParser.print_help
    sys.exit(2)

# TODO right now this can only be run on one subject's baseline phase at a time. this should loop over subjects.
# TODO alternately, accept STDIN argument for a single subject or series of subjects
# for s in subjects:
#    ... initialize PCA, do PCA, output

# subject loop
# subs = [301, 302, 303, 304, 305, 306]
blocks = ['block1', 'block2', 'block3']
print(blocks)
for q in range(312,318):
    SUBJDIR = os.path.join(directory, str(q))
    print(SUBJDIR)
    for b in blocks:
        CURDIR = os.path.join(SUBJDIR, b)

#        e = Exp(expdir=os.path.join(SUBJDIR, b))
        e = Exp(CURDIR)
        e.gather()

        frames = None
        threshhold = 0.020 # threshhold value in s for moving away from acoustic midpoint measure

#        subject = [] # TODO add this in once subject loop is added 
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

            v,m = pm.tier('phone').search(vre, return_match=True)[-1] # return last V = target V
            word = pm.tier('word').label_at(v.center).text
#    print(word)
#    phase.append(a.runvars.phase)
            trial.append(idx)
            tstamp.append(a.timestamp)
            phone.append(v.text)
    
            if args.convert:
                mid, mid_lab, mid_repl = a.frame_at(v.center,missing_val="prev", convert=True)
            else:
                mid, mid_lab, mid_repl = a.frame_at(v.center,missing_val="prev")
#	if args.mask:
#	    pass
#    print(mid) 
    
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

# the following code prints all of the frames as pngs.
#    image_reshape = (416,69)
##    mi = (np.array(mid))
#    mi = mid.reshape(image_reshape)
#    mag = np.max(mi) - np.min(mi)
#    mi = (mi-np.min(mi))/mag*255
#    midmid = np.flipud(e.acquisitions[0].image_converter.as_bmp(mi)) # converter from any frame will work; here we use the first
#    plt.title("PC{:} min/max loadings".format(idx))
#    plt.imshow(midmid, cmap="Greys_r")
#    savepath = "ugh{:}.png".format(idx)
#    plt.savefig(savepath)
                
            frames[idx,:,:] = mid

# # # generate PCA objects # # #

# remove all schwars, if desired (all ER1 outside of learning phase)
#if args.no_er:
#    isnt_er = [f != "ER1" for f in phone]
#    is_learning = [p == "learning" for p in phase]
#    isnt_schwar = [a or b for a,b in zip(isnt_er,is_learning)]


        frames = np.squeeze(frames)
#image_shape = (416,69)
#for frame in frames:

#    print(frame)
#    k = bimp.reshape(image_shape)
#    mag = np.max(k) - np.min(k)
#    k = (k-np.min(k))/mag*255
#    pcn = np.flipud(e.acquisitions[0].image_converter.as_bmp(d)) # converter from any frame will work; here we use the first

#    plt.title("PC{:} min/max loadings".format(mid))
#    plt.imshow(pcn, cmap="Greys_r")
#    savepath = "subj5-pc{:}.pdf".format(mid)
#    plt.savefig(savepath)






        print('here comes the dimension')
        print(np.shape(frames))

        print('here comes trial dimension')
        print(np.shape(trial))
        trial = np.squeeze(np.array(trial))
        print('this is a trial')
        print(trial)
        phone = np.squeeze(np.array(phone))
        print(phone)
        tstamp = np.squeeze(np.array(tstamp))

# remove any indices for all objects generated above where frames have NaN values (due to skipping or otherwise)
        keep_indices = np.where(~np.isnan(frames).any(axis=(1,2)))[0]
        kept_phone = np.array(phone,str)[keep_indices]
        kept_trial = np.array(trial,str)[keep_indices]
        kept_frames = frames[keep_indices]
        kept_tstamp = tstamp[keep_indices]

        (kept_frames)[34][0:200] = 0
        print((kept_frames)[34])
        d = (kept_frames)[34]
        mag = np.max(d) - np.min(d)
        d = (d-np.min(d))/mag*255
        testframe = np.flipud(e.acquisitions[0].image_converter.as_bmp(d)) # converter from any frame will work; here we use the first

        plt.title("Did I turn the bottom row to 0?") #.format?
        plt.imshow(testframe, cmap="Greys_r") 
        savepath = "a_test_frame_lol.pdf" #.format?
        plt.savefig(savepath)

        h = kept_frames.shape[1]
        for i in range(0,(kept_frames.shape[0])):
            (kept_frames)[i][0:210] = 0
            (kept_frames)[i][(h-20):h] = 0

        n_components = 6
        pca = PCA(n_components=n_components)
        print(kept_frames.shape[0]) # this is the number of frames we have in this trial
        print(kept_frames.shape[1]) # this is 416, so I think it's one dimension of the probe.
        print(kept_frames.shape[2]) # this is 69, so I think it's the other dimension of the probe.

# what are the attributes of kept_frames?

        frames_reshaped = kept_frames.reshape([kept_frames.shape[0], kept_frames.shape[1]*kept_frames.shape[2]])
        print('the following is frames_reshaped')
        print(frames_reshaped)
        print(frames_reshaped[0])

        pca.fit(frames_reshaped)
        analysis = pca.transform(frames_reshaped)

        meta_headers = ["trial","timestamp","phone"]
        pc_headers = ["pc"+str(i+1) for i in range(0,n_components)] # determine number of PC columns; changes w.r.t. n_components
        headers = meta_headers + pc_headers

        out_file = os.path.join(CURDIR,args.outfile) # Not sure if this will work

        d = np.row_stack((headers,np.column_stack((kept_trial,kept_tstamp,kept_phone,analysis))))
        np.savetxt(out_file, d, fmt="%s", delimiter =',')

####        print("Data saved. Explained variance ratio of PCs: %s" % str(pca.explained_variance_ratio_))

# TODO save component scores pixel-by-pixel in tabular/csv data such that average loadings by category can be reconstructed


# # # output images describing component min/max loadings # # #

        if args.visualize:
            image_shape = (416,69)

            for n in range(0,n_components):
                d = pca.components_[n].reshape(image_shape)
                print ('look tis d')
                print (d)
                mag = np.max(d) - np.min(d)
                d = (d-np.min(d))/mag*255
                pcn = np.flipud(e.acquisitions[0].image_converter.as_bmp(d)) # converter from any frame will work; here we use the first

                if args.flop:
                    pcn = np.fliplr(pcn)

                plt.title("PC{:} min/max loadings".format(n+1))
                plt.imshow(pcn, cmap="Greys_r") 
                savepath = os.path.join(CURDIR,"subj5-pc{:}.pdf".format(n+1))
                plt.savefig(savepath)

