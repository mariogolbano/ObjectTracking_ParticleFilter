
# Visual Object Tracking Using Particle Filter

## Description

This repository implements a visual object tracking system using a particle filter, based on the work of Bagdanov et al. (2007). The solution has been optimized and extended to improve tracking accuracy by incorporating adaptive models and advanced feature descriptors.

## Project Structure

The project consists of the following files:

- `object_tracking.py`: The main Python script responsible for initializing and running the particle filter to track an object across video frames. It starts by using the ground truth information from the first frame and then reads the video frame by frame to update the filter accordingly. It also computes a per-frame Jaccard Index (JI), which measures the overlap between the predicted and ground truth bounding boxes, and averages it for a final video-level score.

- `particle_filter.py`: A class that implements the particle filter as described in [Bagdanov et al., 2007]. The `__init__()` method initializes the particle filter using the ground truth bounding box from the first frame, while the `update()` method updates the particle filter for each new frame in the video. This file also defines an auxiliary function `computeHSVHistograms()`, which computes a 2D histogram of a bounding box using the H and S channels from the HSV color space.

- `metrics.py`: Defines the function `computeJI()` which calculates the Jaccard Index (JI) between two bounding boxes (ground truth and predicted). The Jaccard Index measures the ratio of the intersection and union areas formed by the two bounding boxes. Its maximum is 1, meaning the bounding boxes are identical, and the minimum is 0, meaning there is no overlap.

- `visualization.py`: Includes two auxiliary functions for visualizing the tracking process:
  - `showBB()`: Displays bounding boxes overlaid on an image to show where the object is being tracked.
  - `showParticles()`: Displays particles overlaid on an image, representing the state of the particle filter.

- `config.py`: The configuration file for the particle filter. It allows setting values for the following parameters:
  - `K`: Number of bins in each dimension of the HS histogram for color tracking.
  - `std_noise`: Standard deviations for noise applied to the state vector.
  - `alpha`: Exponential factor to sharpen the weights distribution.
  - `prediction`: Method used to generate the final prediction of the tracker: two options are currently accepted: 'weighted_avg', which computes the weighted combination of particle states; and 'max', which sets the state to the particle with the maximum weight (the best particle).

- `evaluateSystem.py`: Provides functionality to evaluate the performance of the particle filter on a set of training videos.

- `Instructions.pdf`: Document that describes the baseline implementation of the particle filter and provides guidelines for implementing and evaluating the tracking system.

- `Submitted_Report.pdf`: A report summarizing the optimizations, design decisions, and modifications made to the baseline particle filter.

## Modifications Made in `particle_filter.py`

The `particle_filter.py` file was modified to enhance the robustness and accuracy of the tracking system. The key modifications include:

- **Adaptive Color Model Update**: The color model was adaptively updated to account for changes in the object's appearance over time.

- **Spatial Division of Bounding Box**: The bounding box was divided into subregions, and histograms were computed for each subregion, improving the object representation.

- **Incorporation of Texture (LBP) and HOG Descriptors**: Local Binary Patterns (LBP) and Histograms of Oriented Gradients (HOG) were added to improve robustness, especially in challenging scenarios like motion blur or scale changes.

- **Acceleration Model and Adaptive Bounding Box**: An acceleration model was attempted to predict faster movements and an adaptive bounding box size to adjust scale changes. However, these methods did not improve Jaccard Index and were discarded.

## Evaluation

The performance of the system was evaluated using the Jaccard Index (JI), which measures the overlap between the predicted bounding box and the ground truth. The results obtained were as follows:

| Implementation               | JI Index |
|------------------------------|----------|
| Baseline (Initial)           | 0.334213 |
| Baseline (Corrected)         | 0.497864 |
| Adaptive + Spatial descriptor | 0.483059 |
| Texture descriptor (LBP)     | 0.496133 |
| HOG (Gradient) descriptor    | 0.514160 |
| HOG (Gradient) descriptor (N=300) | 0.516001 |

## Training Videos

To allow training or validating the system on a set of videos, a reduced training set containing 4 videos is provided:

1. Basketball
2. Biker
3. Bolt
4. Skating

For each video, the frames are provided in JPG format, numbered with 4 digits (e.g., "0000.jpg"), and a `nameVideo.mat` file contains the annotations in the following format:

```
gt = (xg, yg, wg, hg)
```

## Usage Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/mariogolbano/ObjectTracking_ParticleFilter.git
   cd ObjectTracking_ParticleFilter
   ```

2. Install the necessary dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script to start tracking:

   ```bash
   python object_tracking.py
   ```

4. To evaluate the system:

   ```bash
   python evaluateSystem.py
   ```

## Contributing

Contributions are welcome. If you want to improve or extend the system, please follow the standard GitHub workflow:

1. Fork the repository.
2. Create a branch for your feature (`git checkout -b feature/new-feature`).
3. Make your changes and commit (`git commit -am 'Add new feature'`).
4. Push the branch (`git push origin feature/new-feature`).
5. Open a Pull Request for review.

