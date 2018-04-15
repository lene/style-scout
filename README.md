# StyleScout
Using neural networks to detect clothes of a style you like

# Done so far
* Download items from the eBay API, turning their most important properties into tags into tags
* Store pickled DB of downloaded items
* training a network (with the InceptionV3 algorithm) on the donloaded images. At the moment it 
  does not perform very well.

# Usage

## First, get yourself some data set

```bash
$ python ebay_download.py [-h] [--verbose] [--items-per-page ITEMS_PER_PAGE]
                        [--page-from PAGE_FROM] [--page-to PAGE_TO]
                        [--save-folder SAVE_FOLDER] [--item-file ITEM_FILE]
                        [--ebay-auth-file EBAY_AUTH_FILE]
                        [--likes-file LIKES_FILE]
                        [--ebay-site_id EBAY_SITE_ID]
                        [--min-valid-tag MIN_VALID_TAG] [--download-images]
                        [--complete-tags-only]
                        [--clean-image-files CLEAN_IMAGE_FILES]
```
Typical usage:
```bash
$ python ebay_download.py -v --items-per-page 100 --page-from 1 --page-to 100 --save-folder data \
    --item-file ebay_items.pickle --download-images --complete-tags-only
```
This tries to download 10000 items in every category that is configured (see `category.py` for the
list of categories), saves their properties to the file `data/ebay_items.pickle` and downloads the
images for the items. Items that are not tagged with all properties needed for training are filtered
out (therefore, realistically  there will be significantly fewer items than 10000 per category).
 
## Second, mark which items you like

For example, you can use `like_items.py` to interactively mark items as liked.
```bash
$ python like_items.py [-h] [--save-folder SAVE_FOLDER] [--start START]
                     [--item-file ITEM_FILE] [--liked-only] [--size SIZE]
```
```bash
$ python like_items.py --item-file ebay_items.pickle --start 100 --size 300

```
This shows you the first four images of all items in the data set, skipping the first 99 items.
Each image is displayed in 300x300 pixels size (this is just the display size and does not affect
the size of the images in the data set). The UI shows buttons to like, unlike or skip each item.

Alternatively, liked items can be recorded in a JSON file which can be passed to `ebay_download.py`
with the command line option `--likes-file`.

## Third, train classifiers on your data set

```bash
$ python train.py [-h] [--verbose] [--save-folder SAVE_FOLDER]
                [--item-file ITEM_FILE] [--min-valid-tag MIN_VALID_TAG]
                [--images-file IMAGES_FILE] [--weights-file WEIGHTS_FILE]
                [--num-epochs NUM_EPOCHS] [--image-size IMAGE_SIZE]
                [--demo DEMO] [--test] [--likes-only] [--category CATEGORY]
                [--batch-size BATCH_SIZE] [--optimizer OPTIMIZER]
                [--type {inception,xception,vgg16,vgg19,resnet50}]
                [--layers LAYERS [LAYERS ...]]
```
Typical usages:
```bash
$ python train.py -v --item-file ebay_items.pickle --category Sandalen --likes-only --num-epochs 10 \
    --weights-file weights_sandalen.hdf5 
```
This trains a classifier on the given data set, for 10 iterations, using only items of the category
"Sandalen" and trying to predict only on whether an item is liked or not. Images are scaled to the 
default size of 299x299 pixels. The resulting weights of the neural network are written to the file 
`weights_sandalen.hdf5`.

```bash
$ python train.py -v --item-file ebay_items.pickle --num-epochs 10 --image-size 400 --batch-size 8 \ 
  --optimizer adam --weights-file weights_400.hdf5 
```
Trains a classifier on all items of all categories, predicting not only the liked status but also
all recorded tags (in the hope of finding cross-correlations). Images are scaled to 400x400, a 
smaller batch size of 8  and the ADAM optimizer is used. 

```bash
$ python train.py -v --item-file ebay_items.pickle --likes-only --type resnet50 --layers 200 100
```
Trains a classifier using the ResNet50 neural network architecture with two additional fully
connected layers of size 200 and 100 

## Finally, predict whether you like or dislike an unknown item

TBD

# Running tests

```bash
$ nosetests tests 
```