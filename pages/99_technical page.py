# in my streamlit page, show to the user all installed python packages
import streamlit as st
import pkg_resources
import pandas as pd
#import plotly.express as px

# Get the list of installed packages
installed_packages = pkg_resources.working_set
installed_packages_list = sorted([f"{i.key}=={i.version}" for i in installed_packages])
# Create a DataFrame from the list of installed packages    
df = pd.DataFrame(installed_packages_list, columns=["Installed Packages"])
# Add a column with the package name and version separately
df["Package Name"] = df["Installed Packages"].apply(lambda x: x.split("==")[0])
df["Version"] = df["Installed Packages"].apply(lambda x: x.split("==")[1])

# now list all the packages in a table
st.title("Installed Python Packages")   
st.write("This is a list of all installed Python packages in the current environment.")
st.write("You can use this information to check the versions of the packages you are using.")

#show on screen with st.write
st.write(df[["Package Name", "Version"]])   


# show a bar chart with the number of packages per version
st.subheader("Number of Packages per Version")



##################################





import joblib
import requests
import traceback
from sklearn.ensemble import RandomForestRegressor
import os



file_id='1i6dUP4QvaAHP3W-wxLc2A9WtpISuU4RU' 
file_url='https://drive.google.com/uc?id=1i6dUP4QvaAHP3W-wxLc2A9WtpISuU4RU'

# we will be using gdown as in https://github.com/wkentaro/gdown
import gdown

destination = "models/reg_rf.pkl"
gdown.download(file_url, destination)

file_size = os.path.getsize(destination)
st.write(f"File size: {file_size} bytes")

# for DEBUG: Display the first 10 characters
#with open(destination, "rb") as file:
#    first_10_chars = file.read(10)
#    st.write(f"First 10 characters: {first_10_chars}")

st.write("Loading the model...")
import joblib
model = joblib.load(destination)
st.success("Model loaded successfully and ready for predictions!")

# TODO: factoriser et appeler tous les modèles "gros"




