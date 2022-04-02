import sys
sys.path.append('..')
import joblib as jbl
import gradio as gr

import pandas as pd
from src.models.train_model import QuantileBasedAnomalyDetection

model = jbl.load('../models/best_model.joblib')

def predict_model(val):
    data_point = pd.DataFrame(data=[val])
    prediction = model.predict(data_point)
    print(prediction)
    if prediction[0] == 1:
        return "Fraudulent"
    else:
        return "Non Fraudulent"

iface = gr.Interface(
    fn=predict_model,
    inputs=[gr.inputs.Slider(-6, 6)],
    outputs=["text"],
)
iface.launch()
