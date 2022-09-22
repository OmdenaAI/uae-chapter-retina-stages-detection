W1: OCT Image preprocessing/Filtering
Tasks leads: @Ege Dolmaci @Muhammed Sahal

The dataset consists a train file with 1000. and 10 images for testing.

https://drive.google.com/drive/folders/1rAgElzDqcopkqhaHFQc6-avkwAsaoJHJ?usp=sharing

‌The gist of preprocessing is to involve techniques that remove unnecessary data from images like noise, extra boundaries, fluid low contrast that affects the blob's appearance.
‌The lower section of the image is not important as a point of interest is the retina nerve fibre on the top which is to be smoothened by applying various preprocessing techniques (mentioned below) for enhanced feature extraction
     * Gray Scale
     * Binarization
     * Morphology 
     * Laplacian of Gaussian 
‌The biggest blob ( Retina fibre ) can be extracted by applying various filters as  mentioned above and trimming off the edges (leaving just the retina fibre for feature extraction).
‌The intention of applying various filters is to facilitate vertical projection on the retina fibre to  creates x and y coordinates (The point of intersection of the retina fibre layer and vertical projections form the coordinates) to describe formation/ Deformation of nerve fibre layer.
‌The coordinates obtained by applying vertical projections can be used for polynomial curve fitting.
