# Industrial Casting Defect Detection

**Deep Learning Image Classification System**  
Built to identify defective vs non-defective casting products using a real-life industrial Kaggle dataset.

## Key Highlights
- Benchmarked a **custom CNN** against **MobileNetV2** transfer-learning model
- Custom CNN architecture: 3 Conv2D blocks (32→64→128 filters) + MaxPooling + Dense(128) + Sigmoid output
- MobileNetV2: Pre-trained ImageNet weights, last 20 layers unfrozen, GlobalAveragePooling + Dropout(0.3)
- Trained for 10 epochs with Adam optimizer and Binary Crossentropy
- Rigorous evaluation using `sklearn.metrics` (classification report + confusion matrix) and Matplotlib/Seaborn plots
- Models saved as `custom_cnn.h5` and `mobilenet_v2.h5`

## Tech Stack
- Python, TensorFlow/Keras, OpenCV, Scikit-learn, Matplotlib, Seaborn
- Data pipeline: `tf.keras.utils.image_dataset_from_directory` + Rescaling(1./255)

## Notebooks & Scripts
- [Untitled0.ipynb](https://github.com/sameerbajpaiiam/casting-defect-detection/blob/main/Untitled0.ipynb)
- [untitled0.py](https://github.com/sameerbajpaiiam/casting-defect-detection/blob/main/untitled0.py) (exported script)

**Repository**: [GitHub](https://github.com/sameerbajpaiiam/casting-defect-detection)
