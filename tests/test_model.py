"""
Tests for the ML Model layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestPreprocessor:
    """Tests for the data preprocessor."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n = 200
        prices = 3800 + np.cumsum(np.random.randn(n) * 10)

        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "symbol": "TCS.NS",
            "open": prices + np.random.randn(n) * 5,
            "high": prices + np.abs(np.random.randn(n) * 10),
            "low": prices - np.abs(np.random.randn(n) * 10),
            "close": prices,
            "volume": np.random.randint(100000, 5000000, n),
        })

    def test_feature_computation(self, sample_data):
        """Test that all features are computed correctly."""
        from src.model.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        features = preprocessor.compute_features(sample_data)

        assert "returns" in features.columns
        assert "volatility_20" in features.columns
        assert "rsi_14" in features.columns
        assert "macd_signal" in features.columns
        assert "volume_ratio" in features.columns
        assert not features["returns"].isna().any()

    def test_fit_transform(self, sample_data):
        """Test fit_transform produces correctly shaped data."""
        from src.model.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        transformed = preprocessor.fit_transform(sample_data)

        assert preprocessor._is_fitted
        assert "close_norm" in transformed.columns
        # close_norm is scaled by _price_scaler to [0, 1]
        # (_feature_scaler only touches raw indicator columns)
        assert transformed["close_norm"].min() >= 0
        assert transformed["close_norm"].max() <= 1

    def test_sequence_creation(self, sample_data):
        """Test that sequence creation produces correct shapes."""
        from src.model.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        transformed = preprocessor.fit_transform(sample_data)
        X, y = preprocessor.create_sequences(transformed)

        assert X.ndim == 3  # (samples, seq_length, features)
        assert y.ndim == 2  # (samples, forecast_horizon)
        assert X.shape[1] == preprocessor.seq_length
        assert X.shape[2] == preprocessor.n_features
        assert y.shape[1] == 1  # forecast_horizon = 1

    def test_inverse_transform(self, sample_data):
        """Test that inverse transform recovers original price scale."""
        from src.model.preprocessor import DataPreprocessor

        preprocessor = DataPreprocessor()
        transformed = preprocessor.fit_transform(sample_data)

        # Get some close_norm values (in [0, 1] from _price_scaler)
        norm_prices = transformed["close_norm"].values[:5]
        # Single-stage inverse via _price_scaler
        recovered = preprocessor.inverse_transform_price(norm_prices)

        # Should be in the same scale as original
        original_range = (sample_data["close"].min(), sample_data["close"].max())
        assert all(
            original_range[0] * 0.9 <= p <= original_range[1] * 1.1
            for p in recovered
        ), f"Recovered prices {recovered} outside expected range {original_range}"


class TestLSTMModel:
    """Tests for the LSTM architecture."""

    def test_model_creation(self):
        """Test that the model can be created."""
        from src.model.lstm import LSTMForecaster

        model = LSTMForecaster(
            input_features=7,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            use_attention=True,
            forecast_horizon=1,
        )

        assert model is not None
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_forward_pass(self):
        """Test that forward pass produces correct output shape."""
        from src.model.lstm import LSTMForecaster

        model = LSTMForecaster(
            input_features=7,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            use_attention=True,
            forecast_horizon=1,
        )
        model.eval()

        # Create random input: (batch=4, seq_len=60, features=7)
        x = torch.randn(4, 60, 7)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"

    def test_forward_with_attention(self):
        """Test forward pass with attention weights return."""
        from src.model.lstm import LSTMForecaster

        model = LSTMForecaster(
            input_features=7,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            use_attention=True,
            forecast_horizon=1,
        )
        model.eval()

        x = torch.randn(2, 60, 7)

        with torch.no_grad():
            output, attn = model(x, return_attention=True)

        assert output.shape == (2, 1)
        assert attn.shape[0] == 2  # batch size
        assert attn.shape[2] == 60  # seq_len
        assert attn.shape[3] == 60  # seq_len

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        from src.model.lstm import LSTMForecaster

        model = LSTMForecaster(
            input_features=7,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            use_attention=True,
            forecast_horizon=1,
        )
        model.train()

        x = torch.randn(4, 60, 7)
        target = torch.randn(4, 1)

        output = model(x)
        loss = torch.nn.MSELoss()(output, target)
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients computed"


class TestTraining:
    """Tests for the training pipeline."""

    def test_training_convergence(self):
        """Test that the model can overfit a small dataset (sanity check)."""
        from src.model.lstm import LSTMForecaster
        from src.model.preprocessor import DataPreprocessor
        from config.settings import config

        # Create simple data
        np.random.seed(42)
        n = 200
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
            "symbol": "TEST",
            "open": prices + np.random.randn(n) * 0.5,
            "high": prices + np.abs(np.random.randn(n)),
            "low": prices - np.abs(np.random.randn(n)),
            "close": prices,
            "volume": np.random.randint(1000, 50000, n),
        })

        preprocessor = DataPreprocessor()
        transformed = preprocessor.fit_transform(df)
        X, y = preprocessor.create_sequences(transformed)

        # Small model for fast test
        model = LSTMForecaster(
            input_features=7,
            hidden_size=32,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            use_attention=False,
            forecast_horizon=1,
        )

        device = "cpu"
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        X_tensor = torch.FloatTensor(X[:32]).to(device)
        y_tensor = torch.FloatTensor(y[:32]).to(device)

        # Train for a few steps
        initial_loss = None
        final_loss = None
        for epoch in range(20):
            model.train()
            pred = model(X_tensor)
            loss = criterion(pred, y_tensor)

            if initial_loss is None:
                initial_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss, (
            f"Model did not converge: initial={initial_loss:.6f}, "
            f"final={final_loss:.6f}"
        )
