ref : Super-BPD: Super Boundary-to-Pixel Direction for Fast Image Segmentation

Thanks to CNNs, semantic segmentation that classifies each pixel into a predefined class category has witnessed significant progress in both accuracy and efficiency.

The BPD is defined on each pixel $p$ as the two-dimensional unit vector pointing from its nearest boundary pixel $B_p$ to $p$.Such BPD encodes the relative position between each pixel and the region boundary. We adopt a CNN to learn such BPD, which is then used to patition the image into super-BPDs.

![illustration of BPD](https://media.arxiv-vanity.com/render-output/3287022/x3.png)

