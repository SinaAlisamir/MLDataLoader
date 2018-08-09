# MLDataLoader

A simple way to load and preprocess big data for machine learning purposes. </br>

Here we use the numpy memory mapping to save each feature in a specific area in the memory. This way we don't have to load all the data into the RAM for training (which is impractical esp. for large datasets and the ones that need preprocessing). </br>
