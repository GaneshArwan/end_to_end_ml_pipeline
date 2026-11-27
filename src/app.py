import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "").strip()
DEFAULT_MODEL_PATH = Path("models/model.joblib")


@st.cache_resource
def load_local_model(model_path: str = str(DEFAULT_MODEL_PATH)) -> Any:
    """Load and cache the serialized scikit-learn pipeline for standalone mode."""
    return joblib.load(model_path)


def predict_local(
    payload: dict[str, float], model_path: str = str(DEFAULT_MODEL_PATH)
) -> tuple[float | None, str | None]:
    """Run in-process prediction using the local joblib artifact."""
    path = Path(model_path)
    if not path.exists():
        return (
            None,
            "Local model artifact not found. Run `python src/train.py` or set `API_URL`.",
        )
    try:
        model = load_local_model(str(path))
        input_df = pd.DataFrame([payload])
        prediction = model.predict(input_df)
        return float(prediction[0]) * 100000, None
    except (ValueError, TypeError, OSError):
        return None, "Local model inference failed."


def request_prediction(
    api_url: str, payload: dict[str, float], timeout: int = 30
) -> tuple[float | None, str | None]:
    """Post house features to the prediction API and return scaled value or error."""
    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        if response.status_code == 200:
            result = response.json()
            # California housing target is in units of $100,000s
            return float(result["predicted_value"]) * 100000, None
        return None, f"API returned status {response.status_code}."
    except requests.exceptions.Timeout:
        return None, "Prediction request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return (
            None,
            "Could not connect to the prediction service. Please verify the API is running.",
        )
    except requests.exceptions.RequestException:
        return None, "Failed to communicate with the prediction API."
    except (ValueError, KeyError, TypeError):
        return None, "Received an unexpected response format from the API."


def get_user_input() -> dict[str, float]:
    """Collect housing feature inputs from the Streamlit sidebar."""
    med_inc = st.sidebar.slider("Median Income (in $10k)", 0.0, 15.0, 3.5)
    house_age = st.sidebar.slider("House Age (years)", 1.0, 100.0, 28.0)
    ave_rooms = st.sidebar.slider("Average Rooms", 1.0, 20.0, 5.0)
    ave_bedrms = st.sidebar.slider("Average Bedrooms", 0.5, 5.0, 1.0)
    population = st.sidebar.slider("Population", 10.0, 10000.0, 1400.0)
    ave_occup = st.sidebar.slider("Average Occupancy", 1.0, 10.0, 3.0)
    latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 35.0)
    longitude = st.sidebar.slider("Longitude", -125.0, -114.0, -119.0)

    return {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }


def main() -> None:
    """Render the Streamlit housing price prediction interface."""
    st.set_page_config(page_title="California Housing Predictor", layout="wide")
    st.title("California Housing Predictor")
    st.write(
        "Predict California block-group median house values from 1990 U.S. Census attributes."
    )
    if API_URL:
        st.caption("Inference mode: FastAPI backend")
    else:
        st.caption("Inference mode: Local model artifact (standalone)")

    st.sidebar.header("Input Features")
    user_input = get_user_input()

    st.subheader("Your Input")
    st.write(pd.DataFrame([user_input]))

    if st.button("Predict"):
        with st.spinner("Calculating..."):
            if API_URL:
                predicted_val, error_msg = request_prediction(API_URL, user_input)
            else:
                predicted_val, error_msg = predict_local(user_input)

            if predicted_val is not None:
                st.success(f"Predicted Median House Value: **${predicted_val:,.2f}**")
            elif error_msg is not None:
                st.error(error_msg)


if __name__ == "__main__":
    main()
