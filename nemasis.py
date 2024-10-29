#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import pandas as pd 
import cv2
import os
from pathlib import Path
from ultralytics import YOLO
from glob import glob
import re

def options():
    parser = argparse.ArgumentParser(description="Nematode egg image processing with YOLOv8 model.")
    parser.add_argument("-i", "--img", help="Target image directory or image (REQUIRED)", required=True)
    parser.add_argument('-w', '--weights', help='Weights file for use with YOLOv8 model (REQUIRED)', required=True)
    parser.add_argument("-o","--output", help="Name of results file. If no file is specified, one will be created from the key file name")
    parser.add_argument("-k", "--key", help="CSV key file to use as output template. If no file is specified, will look for one in target directory. Not used in single-image mode")
    parser.add_argument("-a","--annotated", help="Directory to save annotated image files", required=False)
    parser.add_argument("--conf", help="Confidence cutoff", default=0.6)
    args = parser.parse_args()
    return args

# TODO - maybe rework this from a function to custom argparse.Action() subclasses?
def check_args():
    args = options()
    # basic checks on target file validity
    args.imgpath = Path(args.img)
    if not args.imgpath.exists():
        raise Exception("Target %s is not a valid path" % args.img)
    if args.imgpath.is_file():
        args.img_mode = 'file'
        if not args.imgpath.suffix.lower() in ['.tif','.tiff','.jpg','.jpeg','.png']:
            raise Exception('Target image %s must of type .png, .tif, .tiff, .jpeg, or .jpg' % args.img)
    elif args.imgpath.is_dir():
        args.img_mode = 'dir'
    else:
        raise Exception('Target %s does not appear to be a file or directory.' % args.img)

    # check if subdirectories of format XY00/ exist or if we're running on just a dir of images
    if args.img_mode == 'dir':
        subdirs = sorted(list(args.imgpath.glob('XY[0-9][0-9]/')))
        if len(subdirs) == 0:
            print("No subdirectories of format /XY../ found in specified imgdir, checking for images...")
            potential_images = [x for x in args.imgpath.iterdir() if x.suffix.lower() in ['.tif','.tiff','.jpg','.jpeg','.png']]
            if len(potential_images) == 0:
                raise Exception('No valid images (.png, .tif, .tiff, .jpeg, .jpg) in target folder %s' % args.img)
            else:
                print('%s valid images found' % len(potential_images))
                args.xy_mode = False
                args.subimage_paths = potential_images
        else:
            args.xy_mode = True
            args.subdir_paths = subdirs
        
        # for /XY00/ subdirectories, we require a valid key
        # ensure that either a key is specified, or if a single .csv exists in the target dir, use that
        if args.xy_mode:
            if args.key:
                args.keypath = Path(args.key)
                if not args.keypath.exists():
                    raise Exception('Specified key file does not exist: %s' % args.keypath)
                if args.keypath.suffix != '.csv':
                    raise Exception("Specified key file is not a .csv: %s" % args.keypath)
            else:
                print('Running on /XY00/ subdirectories but no key specified. Looking for key file...')
                potential_keys = list(args.imgpath.glob('*.csv'))
                if len(potential_keys) == 0:
                    raise Exception("No .csv files found in target folder %s, please check directory" % args.img)
                if len(potential_keys) > 1:
                    raise Exception("Multiple .csv files found in target folder %s, please specify which one to use")
                else:
                    args.keypath = potential_keys[0]
                    args.key = str(potential_keys[0])

        # if path to results file is specified, ensure it is .csv
        if args.output:
            args.outpath = Path(args.output)
            if args.outpath.suffix != '.csv':
                raise Exception("Specified output file is not a .csv: %s" % args.outpath)
        else:
            # for XY00 subdirs, name it after the required key file
            # for an image directory, name it after the directory
            if args.xy_mode:
                args.output = '%s_eggcounts.csv' % args.keypath.stem
            else:
                args.output = '%s_eggcounts.csv' % args.imgpath.stem
            args.outpath = Path(args.output)

    # finally, check the target dir to save annotated images in
    if args.annotated:
        args.annotpath = Path(args.annotated)
        if not args.annotpath.exists():
            os.mkdir(args.annotated)
        elif not args.annotpath.is_dir():
            raise Exception("annotated output folder is not a valid directory: %s" % args.annotated)   
    return args

# parse a key file, make sure it all looks correct and can be merged later
def parse_key_file(keypath):
    key = pd.read_csv(keypath)
    # drop potential Unnamed: 0 column if rownames from R were included without col header
    key = key.loc[:, ~key.columns.str.contains('^Unnamed')]
    # for now, will only allow 96-row key files
    # can handle edge cases, but much easier if we just require 96
    if key.shape[0] > 96:
        raise Exception("More than 96 rows found in key. Please check formatting and try again")
    # check if it's got at least one column formatted with what looks like plate positions
    well_columns = []
    for col in key.columns:
        if key[col].dtype.kind == "O":
            if all(key[col].str.fullmatch("[A-H][0-9]{1,2}")):
                well_columns.append(col)
    if len(well_columns) == 0:
        raise Exception("No column found with well positions of format A1/A01/H12/etc.")
    elif len(well_columns) > 1:
        raise Exception("Multiple columns found with well positions of format A1/A01/H12/etc.")
    # add a column named keycol, formatted to match the folder output like _A01
    key["keycol"] = key[well_columns[0]]
    # as the key, it should really be unique and complete, raise exception if not the case
    if any(key["keycol"].isna()):
        raise Exception("There appear to be blank well positions in column %s. Please fix and resubmit." % well_columns[0])    
    if len(set(key["keycol"])) < len(key["keycol"]):
        raise Exception("There appear to be duplicated well positions in the key file. Please fix and resubmit.")
    # if formatted A1, reformat as A01
    key["keycol"] = key["keycol"].apply(lambda x: "_%s%s" % (re.findall("[A-H]",x)[0], re.findall("[0-9]+", x)[0].zfill(2)))
    return key
        
def main():
    args = check_args()
    if args.key:
        key = parse_key_file(str(args.keypath))
    model = YOLO(args.weights)
    # create a couple empty lists for holding results, easier than adding to empty Pandas DF
    tmp_well = []
    tmp_numeggs = []
    tmp_filenames = []
    # single-image mode
    if args.img_mode == 'file':
        img = cv2.imread(str(args.imgpath))
        results = model.predict(img, imgsz = (1920,1440), verbose=False, conf=args.conf)
        result = results[0]
        box_classes = [result.names[int(x)] for x in result.boxes.cls]
        # NOTE - filtering by class is not necessary, but would make this easier to extend to multi-class models
        # e.g. if we want to add hatched, empty eggs, etc
        egg_xy = [x.numpy().astype(np.int32) for i,x in enumerate(result.boxes.xyxy) if box_classes[i] == 'egg']
        print('Target image:\n%s' % str(args.imgpath))
        print('n eggs:\n%s' % len(egg_xy))
        if args.annotated:
            annot = img.copy()
            for xy in egg_xy:
                cv2.rectangle(annot, tuple(xy[0:2]), tuple(xy[2:4]), (0,0,255), 4)
            annot_path = args.annotpath / ('%s_annotated%s' % (args.imgpath.stem, args.imgpath.suffix))
            cv2.imwrite(str(annot_path), annot)
            print('Saving annotations to %s...' % str(annot_path))
    # multi-image mode, runs differently depending on whether you have /XY00/ subdirectories
    elif args.img_mode == 'dir':
        if args.xy_mode:
            for subdir in args.subdir_paths:
                # check that the empty file with well name is present
                well = [x.name for x in subdir.iterdir() if re.match("_[A-H][0-9]{1,2}", x.name)][0]
                if len(well) == 0:
                    raise Exception("No well position file of format _A01 found in subdirectory:\n%s" % subdir)
                # print the XY subdirectory name for tracking purposes
                xy = subdir.name
                print(xy)
                # search for a filename with CH4 in it
                # TODO - confirm with sweetpotato group that the CH4.tif or CH4.jpg will be present in all cases
                candidate_img_paths = list(subdir.glob('*CH4*'))
                # if none or more than one, just skip the folder vs raise exceptions
                if len(candidate_img_paths) == 0:
                    print("No CH4 image found for subdirectory %s" % subdir)
                    continue
                elif len(candidate_img_paths) > 1:
                    print("Multiple CH4 images found in subdirectory %s" % subdir)
                    continue
                impath = candidate_img_paths[0]
                # get the actual output
                img = cv2.imread(str(impath))
                results = model.predict(img, imgsz = (1920,1440), verbose=False, conf=args.conf)
                result = results[0]
                box_classes = [result.names[int(x)] for x in result.boxes.cls]
                egg_xy = [x.numpy().astype(np.int32) for i,x in enumerate(result.boxes.xyxy) if box_classes[i] == 'egg']
                # append relevant output to temporary lists
                tmp_well.append(well)
                tmp_numeggs.append(len(egg_xy))
                tmp_filenames.append(impath.name)
                # annotate and save image if needed
                if args.annotated:
                    annot = img.copy()
                    for xy in egg_xy:
                        cv2.rectangle(annot, tuple(xy[0:2]), tuple(xy[2:4]), (0,0,255), 4)
                    annot_path = args.annotpath / ('%s_annotated%s' % (impath.stem, impath.suffix))
                    cv2.imwrite(str(annot_path), annot)
            # make a CSV to merge with the key
            results = pd.DataFrame({
                "keycol": tmp_well,
                "num_eggs": tmp_numeggs,
                "filename": tmp_filenames,
                "folder": args.img})
            # merge and save
            outdf = key.merge(results, on = "keycol", how = "left")
            outdf = outdf.drop("keycol", axis = 1)
        else:
            # apply the model on each image
            # running model() on the target dir instead of image-by-image would be cleaner
            # but makes saving annotated images more complicated
            # can maybe revisit later
            for impath in sorted(args.subimage_paths):
                img = cv2.imread(str(impath))
                results = model.predict(img, imgsz = (1920,1440), verbose=False, conf= args.conf)
                result = results[0]
                box_classes = [result.names[int(x)] for x in result.boxes.cls]
                egg_xy = [x.numpy().astype(np.int32) for i,x in enumerate(result.boxes.xyxy) if box_classes[i] == 'egg']
                tmp_numeggs.append(len(egg_xy))
                tmp_filenames.append(impath.name)
                # annotate if needed
                if args.annotated:
                    annot = img.copy()
                    for xy in egg_xy:
                        cv2.rectangle(annot, tuple(xy[0:2]), tuple(xy[2:4]), (0,0,255), 4)
                    annot_path = args.annotpath / ('%s_annotated%s' % (impath.stem, impath.suffix))
                    cv2.imwrite(str(annot_path), annot)
            outdf = pd.DataFrame({
                'folder': args.imgpath,
                "filename": tmp_filenames,
                "num_eggs": tmp_numeggs})
        # save final pandas df, print some updates for user
        outdf.sort_values(by='filename', inplace=True)
        outdf.to_csv(str(args.outpath), index=False)
        print('Saving output to %s...' % str(args.outpath))
        if args.annotated:
            print('Saving annotated images to %s...' % str(args.annotpath))
        
if __name__ == '__main__':
    main()