## MambaVision

MambaVision is a hyperspectral image recognition framework built around Mamba-based vision backbones and custom training pipelines.

### Features

- Multi-scale Mamba-based vision models
- Config-driven training via YAML files in `configs`
- Custom schedulers in `scheduler`
- Dataset utilities in `utils`

### Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train a model (example)**

   ```bash
   python trainv2_simple.py --config configs/mambavision_tiny_1k.yaml
   ```

### Project Structure

- `models` – MambaVision model definitions and registry
- `scheduler` – learning rate scheduler implementations
- `utils` – dataset and training utilities
- `configs` – example training configuration files

### License

Specify your preferred license here (e.g., MIT, Apache-2.0).

