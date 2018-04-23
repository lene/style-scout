#!/usr/bin/fish

mkdir -p log

set BATCH_SIZE 2
set ALGOS vgg16 vgg19 xception resnet50 inception

for SIZE in 200 300 400

    for LAYERS in "1024 1024" "1024 1024 512 256"

        for ALGO in $ALGOS
            nice time python train.py -n 10 -v --weights-file {$ALGO}_{$SIZE}_{$LAYERS}_10.hdf5 \
                --item-file data/items_liked_unliked_equal.pickle --image-size {$SIZE} \
                --likes-only --demo 0 --batch-size {$BATCH_SIZE} --optimizer sgd \
                --layer $LAYERS \
                --type $ALGO --test > log/{$ALGO}_{$SIZE}_{$LAYERS}_10.log
            cp {$ALGO}_{$SIZE}_{$LAYERS}_10.hdf5 {$ALGO}_{$SIZE}_{$LAYERS}_20.hdf5
        end

        for ALGO in $ALGOS
            nice time python train.py -n 10 -v --weights-file {$ALGO}_{$SIZE}_{$LAYERS}_20.hdf5 \
                --item-file data/items_liked_unliked_equal.pickle --image-size {$SIZE} \
                --likes-only --demo 0 --batch-size {$BATCH_SIZE}  --optimizer sgd \
                --layer $LAYERS \
                --type $ALGO --test > log/{$ALGO}_{$SIZE}_{$LAYERS}_20.log
            cp {$ALGO}_{$SIZE}_{$LAYERS}_20.hdf5 {$ALGO}_{$SIZE}_{$LAYERS}_50.hdf5
        end

        for ALGO in $ALGOS
            nice time python train.py -n 30 -v --weights-file {$ALGO}_{$SIZE}_{$LAYERS}_50.hdf5 \
                --item-file data/items_liked_unliked_equal.pickle --image-size {$SIZE} \
                --likes-only --demo 0 --batch-size {$BATCH_SIZE} --optimizer sgd \
                --layer $LAYERS \
                --type $ALGO --test > log/{$ALGO}_{$SIZE}_{$LAYERS}_50.log
        end

    end

end