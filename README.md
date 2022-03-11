# cartilage_segmentation


## Getting Started

### Prerequisites

The code runs with Python 3.6.10 , Keras 2.4.3 and Tensorflow 2.2.0.

### Uploading microscopy data

## Running the code

### Training

### Inference 
Run the 'inference' section in 'main.py'. You should choose an option, which automatically sets a group of parameters in 'predict_segmenter_2d()'. Different options correspond to difference training configurations. Every option considers the inference with a series of models trained with different initializations. 

- If TTA is enabled
If TTA is enabled, 20 transformed versions of the input image is provided as input to every model listed in the chosen option. For every transformed input, the inference is performed and progressively accumulated to provide a single fused prediction, which is saved under the name 'X_prediction_fusion_TTA_new_20predictions.npy'. If the computation of the structural biostatistics is enabled, the biostatistics are computed for every transformed input and saved.

- If TTA is disabled
If TTA is disabled, the inference is performed for every model trained with a differerent initialization and every corresponding prediction is saved. If the computation of the structural biostatistics is enabled, the biostatistics are computed for every initialization and saved.

### Fusions

#### Fusion of the predictions using mojority voting
The fusion at the prediction level using majority voting is performed at the inference stage for TTA. The fusion at the prediction level using majority voting is performed using 'save_fusion_ensemble()' when the fusion is performed across different initializations. 

### Generation of the paper results

#### Computation of the biostatistics
Some of the biostatistics can be already computed at the inference stage. If it is not the case, run_compute_bio_stats() allows to do so. 
