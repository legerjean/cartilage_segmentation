# cartilage_segmentation


## Getting Started

### Prerequisites

The code runs with Python 3.6.10 , Keras 2.4.3 and Tensorflow 2.2.0.

### Uploading microscopy data

## Running the code

### Training

### Inference 
Run the 'inference' section in 'main.py'. You should choose an option, which automatically sets a group of parameters in 'predict_segmenter_2d()'. Different options correspond to difference training configurations. Every option considers the inference with a series of models trained with different initializations. 

- If TTA is enabled, 20 transformed versions of the input image is provided as input to every model listed in the chosen option. For every transformed input, the inference is performed and progressively accumulated to provide a single fused prediction, which is saved under the name 'X_prediction_fusion_TTA_new_20predictions.npy'. If the computation of the structural biostatistics is enabled, the biostatistics are computed for every transformed input and saved.

- If TTA is disabled, the inference is performed for every model trained with a differerent initialization and every corresponding prediction is saved. If the computation of the structural biostatistics is enabled, the biostatistics are computed for every initialization and saved.

### Fusions

- Fusion at the prediction level using mojority voting. The fusion at the prediction level using majority voting is performed at the inference stage for TTA. To perform the fusion across different initializations, 'save_fusion_ensemble_full()' is called once and 'save_fusion_ensemble()' five times in 'main.py'. 'save_fusion_ensemble_full()' performs the fusion across 25 initializations. 'save_fusion_ensemble()' performs the fusion across 5 initializations. The structural biostatistics are directly computed after the fusion. 

- Fusion at the biostatistics level using averaging. 

### Computation of the biostatistics

Pas oublier le GT
