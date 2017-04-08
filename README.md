# StyleScout
Using neural networks to detect clothes of a style you like

# Done so far
* Download items from the eBay API, turning their most important properties into tags into tags
* Store pickled DB of downloaded items
* training a network (with the InceptionV3 algorithm) on the donloaded images. At the moment it 
  does not perform very well.

# Rough roadmap/TODO
* only include items which have all necessary tags for a category set
* all save files under one folder instead of having to specify them all
* find a way to not keep the entire data set in memory
* Find an efficient way to like items
  * eBay collections? check out API on them
* check whether including undefined tags as tag:UNDEFINED improves or deteriorates performance
  * check whether to throw out items which do not have all necessary tags set improves performance
* try including material as tag
* check different NN geometries (Xception VGG16/19, ResNet50) and see how they perform compared 
  with InceptionV3
  * also train NNs separately on each category and see how that performs
* try to incorporate text data into the learning process
* Add other eBay sites/countries, internationalize
* Add other services (e.g. Zalando, Kleiderkreisel, etsy, DaWanda, ...)
* World domination!
