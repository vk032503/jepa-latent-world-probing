# JEPA Latent World Probing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Probe and visualize emergent discrete symbols and physical structure in JEPA (Joint Embedding Predictive Architecture) video world model latent representations.

## Overview

Video world models trained with JEPA learn rich spatiotemporal representations by predicting masked regions in latent space rather than reconstructing pixels. This removes the visual verification pathway of generative models, creating a **structural interpretability gap**: the encoder has learned physical structure we can't directly observe.

This toolkit provides comprehensive methods to:

- **Extract physical structure** from JEPA latent representations using linear probing
- **Detect emergent discrete symbols** through clustering and mutual information analysis
- **Measure temporal coherence** to understand how representations track physical entities
- **Visualize latent worlds** with t-SNE projections, transition matrices, and attention maps

## Features

### 🔍 Probing Techniques

- **Linear Probing**: Train linear classifiers to extract physical properties (position, velocity, object identity)
- **Clustering Analysis**: Discover discrete symbol emergence using K-means, DBSCAN, and hierarchical clustering
- **Mutual Information**: Quantify information content between latent representations and physical attributes

### 📊 Analysis Tools

- **Temporal Coherence Metrics**: Measure how well latent representations maintain object identity over time
- **Symbol Transition Analysis**: Visualize state transitions in the discrete latent space
- **Attention Visualization**: Understand what spatial regions the model focuses on

### 🎨 Visualization

- **t-SNE/UMAP Projections**: 2D/3D visualizations of high-dimensional latent spaces
- **Heatmaps**: Attention weights and feature importance
- **Interactive Plots**: Explore latent representations dynamically
- **Video Overlays**: Visualize probing results on original video frames

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jepa-latent-world-probing.git
cd jepa-latent-world-probing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Probing Example

```python
from src.probing import LatentProber
from src.visualization import LatentVisualizer
import numpy as np

# Load your JEPA latent representations
# Shape: (num_samples, latent_dim)
latent_representations = np.load('path/to/latents.npy')
physical_labels = np.load('path/to/labels.npy')  # Ground truth physical properties

# Initialize prober
prober = LatentProber(latent_dim=latent_representations.shape[1])

# Train linear probe for physical properties
probe_results = prober.train_linear_probe(
    latent_representations,
    physical_labels,
    property_name='position'
)

print(f"Probe accuracy: {probe_results['accuracy']:.3f}")
print(f"R² score: {probe_results['r2_score']:.3f}")
```

### 2. Discrete Symbol Detection

```python
from src.symbol_detection import SymbolDetector

# Detect emergent discrete symbols
detector = SymbolDetector(n_symbols=10)
symbol_assignments = detector.fit_predict(latent_representations)

# Analyze symbol quality
metrics = detector.compute_metrics(
    latent_representations,
    symbol_assignments,
    temporal_indices=frame_indices
)

print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
print(f"Temporal coherence: {metrics['temporal_coherence']:.3f}")
```

### 3. Visualization

```python
from src.visualization import LatentVisualizer

visualizer = LatentVisualizer()

# Create t-SNE projection
visualizer.plot_tsne(
    latent_representations,
    labels=symbol_assignments,
    save_path='outputs/tsne_projection.png'
)

# Visualize symbol transitions
visualizer.plot_transition_matrix(
    symbol_assignments,
    temporal_indices=frame_indices,
    save_path='outputs/transitions.png'
)
```

## Complete Workflow Example

```python
from src.pipeline import JEPAProbingPipeline

# Initialize pipeline
pipeline = JEPAProbingPipeline(
    latent_dim=512,
    n_symbols=15,
    output_dir='outputs/'
)

# Run complete analysis
results = pipeline.run_analysis(
    latent_representations=latents,
    physical_labels=labels,
    temporal_indices=frame_ids,
    video_frames=frames  # Optional: for overlay visualizations
)

# Results include:
# - Linear probe accuracies for each physical property
# - Discrete symbol assignments and quality metrics
# - Temporal coherence scores
# - All visualizations saved to output_dir
```

## Project Structure

```
jepa-latent-world-probing/
├── src/
│   ├── __init__.py
│   ├── probing.py              # Linear probing techniques
│   ├── symbol_detection.py     # Discrete symbol emergence detection
│   ├── temporal_analysis.py    # Temporal coherence metrics
│   ├── visualization.py        # Visualization utilities
│   └── pipeline.py             # End-to-end analysis pipeline
├── tests/
│   ├── __init__.py
│   ├── test_probing.py
│   ├── test_symbol_detection.py
│   └── test_visualization.py
├── examples/
│   ├── basic_probing.py
│   ├── symbol_analysis.py
│   └── full_pipeline.py
├── outputs/                    # Generated visualizations and results
├── requirements.txt
├── .env.example
└── README.md
```

## Methodology

### Linear Probing

Linear probing trains simple linear classifiers on frozen latent representations to predict physical properties. High accuracy indicates that the information is linearly accessible in the latent space.

**Supported Properties:**
- Object position (x, y coordinates)
- Velocity (vx, vy)
- Object identity/class
- Physical attributes (size, color, etc.)

### Discrete Symbol Detection

We use multiple clustering algorithms to identify emergent discrete symbols:

1. **K-means**: Fast, assumes spherical clusters
2. **DBSCAN**: Density-based, discovers arbitrary shapes
3. **Hierarchical**: Reveals symbol hierarchy

**Quality Metrics:**
- Silhouette score: Cluster separation quality
- Temporal coherence: Symbol stability over time
- Mutual information: Information content

### Temporal Coherence

Measures how well latent representations maintain object identity across frames:

```
coherence = 1 - (symbol_changes / total_transitions)
```

High coherence indicates the model has learned stable object representations.

## Advanced Usage

### Custom Physical Properties

```python
# Define custom property extractor
def extract_custom_property(video_frame, object_id):
    # Your custom logic here
    return property_value

# Use with prober
prober.add_custom_property('my_property', extract_custom_property)
```

### Multi-Modal Probing

```python
# Probe multiple properties simultaneously
multi_results = prober.train_multi_probe(
    latent_representations,
    properties={
        'position': position_labels,
        'velocity': velocity_labels,
        'identity': identity_labels
    }
)
```

### Attention Visualization

```python
# Visualize spatial attention patterns
visualizer.plot_attention_heatmap(
    attention_weights,  # From JEPA encoder
    video_frame,
    save_path='outputs/attention.png'
)
```

## Research Context

This toolkit implements methods from:

**"Probing the Latent World: Emergent Discrete Symbols and Physical Structure in Latent Representations"**  
arXiv:2603.20327

Key findings:
- JEPA models learn linearly accessible physical structure
- Discrete symbols emerge naturally in latent space
- Temporal coherence correlates with downstream task performance

## Performance Considerations

- **Memory**: For large datasets, use batch processing
- **Speed**: Linear probing is fast; clustering scales O(n²) for some algorithms
- **GPU**: Visualization can leverage GPU for large-scale t-SNE

```python
# Enable batch processing for large datasets
pipeline = JEPAProbingPipeline(batch_size=1000)
```

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@article{jepa-latent-probing-2024,
  title={Probing the Latent World: Emergent Discrete Symbols and Physical Structure in Latent Representations},
  journal={arXiv preprint arXiv:2603.20327},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- JEPA architecture from Meta AI Research
- Inspired by interpretability research in self-supervised learning
- Built with PyTorch, scikit-learn, and matplotlib

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy probing! 🔍**