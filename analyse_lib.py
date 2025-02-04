import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

no_of_candidates = json.load(open("./Data/no_of_candidates.json"))

class BidirectionalPredictor:
    def __init__(self, degree=10):
        """
        Initialize predictors for both directions
        """
        self.degree = degree
        # Predictor for marks -> ranks
        self.marks_to_ranks_model = self._create_model()
        # Predictor for ranks -> marks
        self.ranks_to_marks_model = self._create_model()

    def _create_model(self):
        """Helper function to create a model with scaler"""
        return {
            'poly_features': PolynomialFeatures(degree=self.degree),
            'model': LinearRegression(),
            'scaler': MinMaxScaler()
        }

    def train(self, marks, ranks):
        """Train both directional models"""
        # Train marks -> ranks model
        self._train_single_direction(
            self.marks_to_ranks_model,
            np.array(marks).reshape(-1, 1),
            np.array(ranks).reshape(-1, 1)
        )

        # Train ranks -> marks model
        self._train_single_direction(
            self.ranks_to_marks_model,
            np.array(ranks).reshape(-1, 1),
            np.array(marks).reshape(-1, 1)
        )

    def _train_single_direction(self, model_dict, X, y):
        """Helper function to train a single direction"""
        X_scaled = model_dict['scaler'].fit_transform(X)
        X_poly = model_dict['poly_features'].fit_transform(X_scaled)
        model_dict['model'].fit(X_poly, y)

    def predict_rank(self, marks):
        """Predict ranks given marks"""
        return self._predict_single_direction(
            self.marks_to_ranks_model,
            np.array(marks).reshape(-1, 1)
        )

    def predict_marks(self, ranks):
        """Predict marks given ranks"""
        return self._predict_single_direction(
            self.ranks_to_marks_model,
            np.array(ranks).reshape(-1, 1)
        )

    def _predict_single_direction(self, model_dict, X):
        """Helper function for prediction"""
        assert 'scaler' in model_dict, "Key 'scaler' not found in model_dict"
        assert 'poly_features' in model_dict, "Key 'poly_features' not found in model_dict"
        assert 'model' in model_dict, "Key 'model' not found in model_dict"

        X_scaled = model_dict['scaler'].transform(X)
        X_poly = model_dict['poly_features'].transform(X_scaled)
        predictions = model_dict['model'].predict(X_poly)
        return np.round(predictions.flatten())

    def plot_both_directions(self, marks, ranks):
        """Plot both prediction directions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot marks -> ranks
        ax1.scatter(marks, ranks, color='blue', label='Actual Data')
        X_smooth = np.linspace(min(marks), max(marks), 300)
        y_smooth = self.predict_rank(X_smooth)
        ax1.plot(X_smooth, y_smooth, color='red', label='Prediction')
        ax1.set_xlabel('Marks')
        ax1.set_ylabel('Rank')
        ax1.set_title('Marks → Rank Prediction')
        ax1.legend()
        ax1.grid(True)

        # Plot ranks -> marks
        ax2.scatter(ranks, marks, color='blue', label='Actual Data')
        X_smooth = np.linspace(min(ranks), max(ranks), 300)
        y_smooth = self.predict_marks(X_smooth)
        ax2.plot(X_smooth, y_smooth, color='red', label='Prediction')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Marks')
        ax2.set_title('Rank → Marks Prediction')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

class Marks_vs_Rank_Analyser:
    def __init__(self, year):
        self.year = year
        marks_vs_rank = pd.read_csv(f"./Data/{self.year}/{self.year}_Marks_vs_percentile.csv", index_col=[0], header=[0])
        marks_vs_rank.index.name = "Percentile"
        exam_dates = marks_vs_rank.columns
        marks_vs_rank["Ranks"] = (100 - marks_vs_rank.index)/100*no_of_candidates["2024"]
        marks_vs_rank["Max"] = marks_vs_rank[exam_dates].max(axis=1)
        marks_vs_rank["Min"] = marks_vs_rank[exam_dates].min(axis=1)
        marks_vs_rank["Avg"] = marks_vs_rank[exam_dates].mean(axis=1)
        marks_vs_rank["Median"] = marks_vs_rank[exam_dates].median(axis=1)

        # Minimum
        self.min_score_predictor = BidirectionalPredictor()
        self.min_score_predictor.train(marks_vs_rank["Min"].to_numpy(), marks_vs_rank["Ranks"].to_numpy())

        # Maximum
        self.max_score_predictor = BidirectionalPredictor()
        self.max_score_predictor.train(marks_vs_rank["Max"].to_numpy(), marks_vs_rank["Ranks"].to_numpy())

        # Average
        self.avg_score_predictor = BidirectionalPredictor()
        self.avg_score_predictor.train(marks_vs_rank["Avg"].to_numpy(), marks_vs_rank["Ranks"].to_numpy())

        # Median
        self.median_score_predictor = BidirectionalPredictor()
        self.median_score_predictor.train(marks_vs_rank["Median"].to_numpy(), marks_vs_rank["Ranks"].to_numpy())

    def predict_marks(self, ranks):
        """Predict marks given ranks"""
        return self.min_score_predictor.predict_marks(ranks), self.max_score_predictor.predict_marks(ranks), self.avg_score_predictor.predict_marks(ranks), self.median_score_predictor.predict_marks(ranks)

    def predict_ranks(self, marks):
        """Predict marks given ranks"""
        return self.min_score_predictor.predict_rank(marks), self.median_score_predictor.predict_rank(marks)