import streamlit as st
import time
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

st.markdown("# CO2 Demo")
st.sidebar.header("CO2 Demo")
st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

import os
print(os.getcwd())


image_path = "images/CorrectionsEmpattement.jpg"  # Adjust path relative to your file
image = Image.open(image_path)
st.image(image, caption="Outliers d'empattements comparés à la masse.")

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")

