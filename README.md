# Extreme Learning Machine (ELM) with PCA for Image Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art implementation of an Extreme Learning Machine (ELM) with PCA dimensionality reduction, designed for robust image classification. This repository features advanced data augmentation, neural network visualization tools, and systematic hyperparameter optimization.

## Key Features ✨

- **PCA-Enhanced ELM Architecture**
  - Whitened PCA for noise-resistant feature extraction
  - He/Xavier weight initialization schemes
  - Regularized pseudoinverse optimization (L2 penalty)
  
- **Advanced Data Augmentation**
  - Geometric: ±30° rotation, random flips, 70-90% cropping
  - Photometric: Contrast variation (0.5-1.5x), Gaussian noise (σ=25)
  - On-the-fly augmentation during training

- **Diagnostic Visualization Suite**
  - Node importance network graphs
  - Weight distribution heatmaps
  - Receptive field visualizations

- **Optimization Framework**
  - Grid search over critical parameters
  - Stratified validation splits
  - Automatic model selection


## Installation 🛠️

### Requirements
- Python 3.8+
- GPU recommended for large datasets

### Setup
```bash
git clone https://github.com/yourusername/elm-pca-image-classification.git
cd elm-pca-image-classification

# Create virtual environment
python -m venv elm_env
source elm_env/bin/activate  # Linux/MacOS
# elm_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage 🚀

### Dataset Structure
```
KTH_train/
    class1/
        img1.jpg
        img2.jpg
    class2/
        ...
KTH_test/
    ... (same structure)
```

### Training & Evaluation
```python
# Main execution with default parameters
python main.py

# Custom training (example)
from EnhancedELM import EnhancedELM
from ImageProcessor import ImageProcessor

processor = ImageProcessor(input_shape=(200, 200))
model = EnhancedELM(hidden_units=2000, pca_var=0.95)
model.fit(X_train, y_train)
accuracy = model.evaluate(X_test, y_test)
```

### Visualization
```python
visualizer = ELMVisualizer(trained_model)
visualizer.plot_node_network()        # Interactive node graph
visualizer.plot_weight_heatmap()      # Class-node relationships
visualizer.plot_receptive_fields()    # Feature detectors
```

## Hyperparameter Tuning ⚙️

| Parameter          | Recommended Range | Effect                           |
|--------------------|-------------------|----------------------------------|
| `hidden_units`     | 500-3000          | Model capacity/complexity        |
| `pca_var`          | 0.85-0.95         | Feature compression level        |
| `reg_param`        | 1e-7-1e-5         | Overfitting control              |
| `activation`       | relu/tanh         | Non-linear transformation        |

```python
# Example grid search configuration
param_grid = {
    'pca_var': [0.85, 0.90, 0.95],
    'hidden_units': [1000, 2000, 3000],
    'reg_param': [1e-7, 1e-6, 1e-5],
    'activation': ['relu', 'tanh']
}
```


## Contributing 🤝

1. Fork the repository
2. Create feature branch (`git checkout -b feature`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push branch (`git push origin feature`)
5. Open Pull Request

## References 📚

1. Huang, G.B., et al. (2006). ["Extreme Learning Machine: Theory and Applications"](https://doi.org/10.1016/j.neucom.2005.12.126)
2. Pedregosa et al. (2011). ["Scikit-learn: Machine Learning in Python"](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
3. KTH Action Dataset: [Official Page](https://www.csc.kth.se/cvap/actions/)