#!/bin/bash

### Script to create train and val directories for image net data
### No test set for now

train_size="0.7"
data_dir="/Users/alex/Desktop/Projects/wwf_sharks/data/"
main_folder="val"
train_folder="train"
min_imgs="400"

cd $data_dir
cd $main_folder

# use while instead of for, because dir names have spaces.
find . -type d | while read dir; do

    #cd $d/
    num_imgs=$(ls -1 "$dir" | wc -l)

    if [$num_imgs -gt $min_imgs ]
    then
        echo "$dir has $num_imgs images"

        # compute number of train images
        float=$(bc<<<"$train_size * $num_imgs")
        int=${float%.*}

        new_dir=$"../../$train_folder/$dir"
        echo "We will take $int images for training and move them to $new_dir "

        #echo "$int"
        #f=$(ls $dir | head -500)
        f=$(ls "$dir" | head -"$int")
        cd "$dir"
        for file in $f; do
            mv $file "$new_dir"
        done
        cd ../
        # make class directory in train
        #mkdir "../$train_folder/$dir"
        #ls | head -$int | xargs -I{} mv {} $new_dir
        #for file in $(ls -p | grep -v / | tail -$int);do
            #cd $dir
        #    echo $file
            #echo "../../$train_folder/$dir"
            #mv $file ../../$train_folder/$dir
            #cd ../
        #done

        # move train images to train folder
    fi

    #ls "$dir" | head -1
    #cd ../
done
