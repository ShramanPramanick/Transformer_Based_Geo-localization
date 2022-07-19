# Transformer_Based_Geo-Localization
This is the repository for ECCV 2022 paper titled: "Where in the World is this Image? Transformer-based Geo-localization in the Wild". More details and links to the training dataset and corresponding semantic segmentation maps will be available soon.

## Download Training Annotations
Download the annotations of mp16 training set and keep them in the resources folder.
```
cd resources/
wget http://www.cis.jhu.edu/~shraman/mp16_places365.csv
wget http://www.cis.jhu.edu/~shraman/mp16_places365_mapping_h3.json
```

## Download Evaluation Datasets
Download the RGB images of the three evaluation datasets and their corresponding semantic maps, and untar them into resources folder.

**Im2GPS dataset**
```
wget http://www.cis.jhu.edu/~shraman/im2gps_rgb_images.tar.gz
wget http://www.cis.jhu.edu/~shraman/im2gps_seg_images_PNG.tar.gz
```
**Im2GPS3k dataset**
```
wget http://www.cis.jhu.edu/~shraman/im2gps3k_rgb_images.tar.gz
wget http://www.cis.jhu.edu/~shraman/im2gps3k_seg_images_PNG.tar.gz
```
**YFCC4k dataset**
```
wget http://www.cis.jhu.edu/~shraman/yfcc4k_rgb_images.tar.gz
wget http://www.cis.jhu.edu/~shraman/yfcc4k_seg_images_PNG.tar.gz
```


