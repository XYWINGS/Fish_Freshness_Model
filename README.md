```markdown
# Fish Freshness Classification System

A deep learning system that classifies fish freshness based on eye and gill images, distinguishing between "fresh" and "non-fresh" states.

## Features

- **Multi-modal analysis**: Combines eye and gill features for accurate classification
- **Advanced architecture**: Uses ResNet18 + Vision Transformer with attention mechanisms
- **Smart pre-filtering**: Automatically rejects irrelevant images (text, objects, etc.)
- **REST API**: Easy integration via Flask web service

## Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3 (for GPU acceleration)
- Other dependencies in `requirements.txt`

## Installation

1. Clone the repository:

2. Set up the environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

3. Download and extract the model files:
```bash
unzip model/multi_attribute_fish_model_novel2_part1.zip -d model/
```

## Usage

### Running the API Server
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Making Predictions

Send a POST request with an image file:

```bash
curl -X POST -F "file=@test_fish.jpg" http://localhost:5000/predict
```

Example response:
```json
{
    "confidence": 0.92,
    "predicted_class": "fresh"
}
```


## Model Details

**Architecture**:
- Hybrid CNN (ResNet18) + Vision Transformer
- Attention mechanisms for eye/gill feature weighting
- Multi-modal feature fusion

**Performance**:
- Accuracy: 94.2% on validation set
- Inference time: ~120ms (on NVIDIA T4 GPU)
