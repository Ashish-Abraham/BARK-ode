import json
from io import BytesIO
from PIL import Image
import os


import streamlit as st
import pandas as pd
import numpy as np

from resnet_model import ResnetModel


@st.cache()
def load_model(path: str = 'models/trained_model_resnet50.pt') -> ResnetModel:
    """Retrieves the trained model and maps it to the CPU by default,
    can also specify GPU here."""
    model = ResnetModel(path_to_pretrained_model=path)
    return model