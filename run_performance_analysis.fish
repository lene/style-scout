#!/usr/bin/fish

mkdir -p log

set BATCH_SIZE 2
set ALGOS vgg16 vgg19 xception resnet50 inception

for SIZE in 200 300 400

    for ALGO in $ALGOS
        nice time python train.py -n 10 -v --weights-file {$ALGO}_{$SIZE}_10.hdf5 \
            --item-file data/items_liked_unliked_equal.pickle --image-size {$SIZE} \
            --likes-only --demo 0 --batch-size {$BATCH_SIZE} --optimizer sgd \
            --layer 1024 --layer 1024 \
            --type $ALGO --test > log/{$ALGO}_{$SIZE}_10.log
        cp {$ALGO}_{$SIZE}_10.hdf5 {$ALGO}_{$SIZE}_20.hdf5
    end

    for ALGO in $ALGOS
        nice time python train.py -n 10 -v --weights-file {$ALGO}_{$SIZE}_10.hdf5 \
            --item-file data/items_liked_unliked_equal.pickle --image-size {$SIZE} \
            --likes-only --demo 0 --batch-size {$BATCH_SIZE} --optimizer sgd \
            --layer 1024 --layer 1024 --layer 512 --layer 256 \
            --type $ALGO --test > log/{$ALGO}_{$SIZE}_10.log
        cp {$ALGO}_{$SIZE}_10.hdf5 {$ALGO}_{$SIZE}_20.hdf5
    end

    for ALGO in $ALGOS
        nice time python train.py -n 10 -v --weights-file {$ALGO}_{$SIZE}_20.hdf5 \
            --item-file data/items_liked_unliked_equal.pickle --image-size {$SIZE} \
            --likes-only --demo 0 --batch-size {$BATCH_SIZE}  --optimizer sgd \
            --layer 1024 --layer 1024 \
            --type $ALGO --test > log/{$ALGO}_{$SIZE}_20.log
        cp {$ALGO}_{$SIZE}_20.hdf5 {$ALGO}_{$SIZE}_50.hdf5
    end

    for ALGO in $ALGOS
        nice time python train.py -n 10 -v --weights-file {$ALGO}_{$SIZE}_20.hdf5 \
            --item-file data/items_liked_unliked_equal.pickle --image-size {$SIZE} \
            --likes-only --demo 0 --batch-size {$BATCH_SIZE}  --optimizer sgd \
            --layer 1024 --layer 1024 --layer 512 --layer 256 \
            --type $ALGO --test > log/{$ALGO}_{$SIZE}_20.log
        cp {$ALGO}_{$SIZE}_20.hdf5 {$ALGO}_{$SIZE}_50.hdf5
    end

    for ALGO in $ALGOS
        nice time python train.py -n 30 -v --weights-file {$ALGO}_{$SIZE}_50.hdf5 \
            --item-file data/items_liked_unliked_equal.pickle --image-size {$SIZE} \
            --likes-only --demo 0 --batch-size {$BATCH_SIZE} --optimizer sgd \
            --layer 1024 --layer 1024
            --type $ALGO --test > log/{$ALGO}_{$SIZE}_50.log
    end

    for ALGO in $ALGOS
        nice time python train.py -n 30 -v --weights-file {$ALGO}_{$SIZE}_50.hdf5 \
            --item-file data/items_liked_unliked_equal.pickle --image-size {$SIZE} \
            --likes-only --demo 0 --batch-size {$BATCH_SIZE} --optimizer sgd \
            --layer 1024 --layer 1024 --layer 512 --layer 256 \
            --type $ALGO --test > log/{$ALGO}_{$SIZE}_50.log
    end

end