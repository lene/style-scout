Performance
===========

NVidia Quadro M1000M, 2GB VRAM

817 liked items, 817 random unliked items, image size 300x300

|Algorithm   |Extra Layers  |Time (10 epochs)       |Test set acc. (10 epochs) |(20 epochs)|(50 epochs)| 
|:-----------|:-------------|:-------------------------|:----------------------|:----------|:----------|
|VGG16       |2x1024        |2:42:12 wall,  6527.80s user, 1873.77s sys|0.6757 |0.6486 |0.6359 |
|VGG19       |2x1024        |3:22:50 wall,  7856.31s user, 2342.34s sys|0.6734 |0.6201 |       |
|Xception    |2x1024        |3:14:59 wall, 11755.85s user, 2448.42s sys|0.6652 |0.6547 |       |
|Resnet50    |2x1024        |2:11:18 wall,  9060.37s user, 1920.83s sys|0.6149 |       |       |
|Inception   |2x1024        |1:47:40 wall,  8793.54s user, 1606.16s sys|0.5901 |       |       |
