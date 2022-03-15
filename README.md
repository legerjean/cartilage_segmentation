# cartilage_segmentation


## Getting Started

### Prerequisites

The code runs with Python 3.6.10 , Keras 2.4.3 and Tensorflow 2.2.0.

### Uploading microscopy data
Raw JPEG images from 12 murine bone-to-Achilles tendon interfaces and corresponding BMP manual annotations are converted in numpy arrays using 'generate_data.py'. The 3D images and manual annotations are cropped around the mineralized cartilage region of interest and saved in separated files. For training purposes, 100 slices of each 3D image are selected and concatenated to build a single numpy array with dimensions (1200, 1024, 1024). The same is done with the manual annotations.

## Running the code

### Training
Models with different initializations are trained using different training configurations in the 'training' section in 'main.py'. 

### Inference 
Run the 'inference' section in 'main.py'. You should choose an option, which automatically sets a group of parameters in 'predict_segmenter_2d()'. Different options correspond to difference training configurations. Every option considers the inference with a series of models trained with different initializations. 

- If TTA is enabled, 20 transformed versions of the input image are provided as input to every model listed in the chosen option. For every transformed input, the inference is performed and progressively accumulated to provide a single fused prediction, which is saved under the name 'X_prediction_fusion_TTA_new_20predictions.npy'. If the computation of the characteristics is enabled, the characteristics are computed for every transformed input and saved.

- If TTA is disabled, the inference is performed for every model trained with a differerent initialization and every corresponding prediction is saved. If the computation of the characteristics is enabled, the characteristics are computed for every initialization and saved.

### Fusions

- Fusion at the prediction level using majority voting. The fusion at the prediction level using majority voting is performed at the inference stage for TTA. The characteristics are computed using 'majority_voting_TTA_bio_stats()'. To perform the fusion across different initializations, 'save_fusion_ensemble_full()' is called once and 'save_fusion_ensemble()' five times in 'main.py'. 'save_fusion_ensemble_full()' performs the fusion across 20 initializations. 'save_fusion_ensemble()' performs the fusion across 5 initializations. The characteristics are directly computed after the fusion. 

- Fusion at the biological characteristic level using averaging. The fusion at the characteristic level using averaging is performed by 'average_TTA_bio_stats()' and 'average_ensemble_bio_stats()' in 'main.py' for the TTA and ensembling strategies, respectively.

### Getting the results of the paper

Run the remaining lines in 'main.py' in order to compute the biological characteristics using the annotations, compute the calibration, compute the uncertainty, and plot the figures and tables shown in the paper.
