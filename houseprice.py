# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models

st.title("🏡 House Price Prediction (TensorFlow)")

st.write("Upload a CSV dataset with a `price` column (target) and numeric features like `beds`, `baths`, `size`, etc.")

# File uploader
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    if "price" not in df.columns:
        st.error("Dataset must have a column named 'price'")
    else:
        # Split data
        X = df.drop("price", axis=1).select_dtypes(include=["number"]).fillna(0)
        y = df["price"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build TensorFlow model
        model = models.Sequential([
            layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
            layers.Dense(32, activation="relu"),
            layers.Dense(1)  # Regression output
        ])

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=50,
            batch_size=32,
            verbose=0
        )

        # Evaluate
        preds = model.predict(X_test).flatten()
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        st.write(f"📊 Model trained with TensorFlow! RMSE on test set: **{rmse:.2f}**")

        # Prediction form
        st.subheader("🔮 Predict New House Price")
        input_data = {}
        for col in X.columns:
            val = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
            input_data[col] = val

        if st.button("Predict Price"):
            input_df = pd.DataFrame([input_data])
            pred_price = model.predict(input_df)[0][0]
            st.success(f"Estimated House Price: ${pred_price:,.2f}")