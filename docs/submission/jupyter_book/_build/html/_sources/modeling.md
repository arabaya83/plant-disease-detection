# Modeling

Three models were built and compared:

1. Simple CNN baseline (`ml/src/models/cnn_baseline.py`)
2. MobileNetV2 baseline (`ml/src/models/mobilenet_baseline.py`)
3. Hybrid model (`ml/src/models/hybrid_model.py`)

The hybrid model fuses transfer-learned MobileNet features with a custom residual branch via concatenation.
