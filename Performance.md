Performance
===========

NVidia Quadro M1000M, 2GB VRAM

817 liked items, 817 random unliked items, image size 300x300, batch size 2

|Algorithm   |Extra Layers      |Time (10 epochs)          |Test set acc. (10 epochs) |(20 epochs)|(50 epochs)| 
|:-----------|-----------------:|-------------------------:|:----------------------|:----------|:----------|
|VGG16       |1024              |2:42:12 wall,  6527.80user, 1873.77sys|0.6757     |0.6486     |0.6359     |
|VGG19       |1024              |3:22:50 wall,  7856.31user, 2342.34sys|0.6734     |0.6201     |0.6547     |
|Xception    |1024              |3:14:59 wall, 11755.85user, 2448.42sys|0.6652     |0.6547     |0.6599     |
|Resnet50    |1024              |2:11:18 wall,  9060.37user, 1920.83sys|0.6149     |       |       |
|Inception   |1024              |1:47:40 wall,  8793.54user, 1606.16sys|0.5901     |       |       |
|VGG16       |1024,1024,512,256 |2:44:00 wall,  6483.41user, 1988.42sys|0.5571     |0.6344     | |
|VGG19       |1024,1024,512,256 |3:22:55 wall,  7491.32user, 2372.79sys|0.6614     | | |
|Xception    |1024,1024,512,256 |3:25:05 wall, 11923.87user, 2982.10sys|0.5818     | | |
|Resnet50    |1024,1024,512,256 |2:12:18 wall,  9186.89user, 1942.89sys|0.5961     |       |       |
|Inception   |1024,1024,512,256 |1:49:01 wall,  8739.31user, 1839.92sys|0.5225     |       |       |
