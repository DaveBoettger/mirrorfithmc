import sys
import argparse
import pymc3 as pm
import numpy as np
try:
    import mirrorfithmc.mirrorfithmc as mf
except:
    import mirrorfithmc as mf
import matplotlib.pyplot as plt

dist_limit = 10 #match to points within 10 mm

def has_duplicate_label(ds):
    lbls = ds.labels
    if len(set(lbls)) != len(lbls):
        return True
    else:
        return False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to the dataset file you want to rename")
    parser.add_argument("--template", help="Path to the template file you want to rename with", action="append")
    parser.add_argument("--outfile", help="Path to the file to output", default='renamed_output.txt')
    args = parser.parse_args()

    rename_ds = mf.Dataset(from_file=args.dataset)
    rename_vals = []

    if has_duplicate_label(rename_ds):
        print(f'Dataset {rename_ds.label} has duplicate labels. If this is true after renaming it will not be saved.')

    for template in args.template:
        this_temp = mf.Dataset(from_file=template)
        if has_duplicate_label(this_temp):
            print(f'Template {this_temp.name} has duplicate labels. Skipping.')
            continue

        with mf.AlignDatasets(ds1=this_temp, ds2=rename_ds, use_marker='CODE') as model:
            maptrans = pm.find_MAP(model=model)
            vals = ['tx','ty','tz','rx','ry','rz','s']
            tdict = {}
            for val in vals:
                for key in maptrans:
                    if val in key:
                        tdict[val] = maptrans[key]
            trans = mf.TheanoTransform(trans=tdict)
            renamet = rename_ds.to_tensors()
            renametprime = trans*renamet
            rename_pos = renametprime.pos.eval()
            for point in this_temp.values():
                this_vec_dists = rename_pos.T-point.pos
                this_dists = np.linalg.norm(this_vec_dists,axis=1)
                minidx = np.argsort(this_dists)
                min_dist = this_dists[minidx][0]
                next_dist = this_dists[minidx][1] #Do we want to do anything with this? 
                if min_dist < dist_limit:
                    if next_dist < dist_limit:
                        print(f'Found more than one close point (at {min_dist:.2f} and {next_dist:.2f} mm) for {point.label}, skipping. Consider adjusting dist_limit.')
                        continue    

                    rename_vals.append((minidx[0], point.label))

    rename_labels = rename_ds.labels
    renamed = 0
    for v in rename_vals:
        curr_label=rename_labels[v[0]]
        new_label = v[1]
        if 'CODE' in curr_label or 'CODE' in new_label:
            if curr_label == new_label:
                continue
            else:
                raise ValueError(f'Trying to overwrite {curr_label} with {new_label} - there was probably a problem with alignment. Aborting.')
        #Rename the point in place, which doesn't change the dictionary key
        #but will be written out to the file when we save it.
        rename_ds[curr_label].label = new_label
        renamed+=1 
    
    if has_duplicate_label(rename_ds):
        print(f'Dataset {rename_ds.label} has duplicate labels and will not be saved.')
        quit()
    
    rename_ds.write_data_file(args.outfile)
