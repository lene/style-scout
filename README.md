# StyleScout
Using neural networks to detect clothes of a style you like

# Rough roadmap
* Turn properties of downloaded items into tags
** only use tags which have been used more than n times, because some are free text
** tags which are not set need a '<Category>:None' tag
* Store DB of downloaded items (pickle should do nicely)
* Download more (*many* more) items
** round-robin through the selected categories
* Train NN against extracted tags
* Find an efficient way to like items
* Add "like" property to tags and retrain NN
* Add other eBay sites/countries, internationalize
* Add other services (e.g. Zalando, Kleiderkreisel, ...)
* World domination!