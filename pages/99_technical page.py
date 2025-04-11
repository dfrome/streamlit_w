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

# Provide the public URL and convert it to a direct download link
public_url = "https://drive.google.com/file/d/1i6dUP4QvaAHP3W-wxLc2A9WtpISuU4RU/view?usp=sharing"
file_id = public_url.split("/d/")[1].split("/")[0]  # Extract the file ID
download_url = f"https://drive.google.com/uc?id={file_id}&export=download"

st.write("### Downloading the model...")
st.write(f"**Model URL:** {download_url}")  

# Download the file and cache it
@st.cache_data
def download_model():
    response = requests.get(download_url)
    with open("reg_rf.pkl", "wb") as file:
        file.write(response.content)
    return "reg_rf.pkl"

# Load the model
try:
    model_path = download_model()
    model = joblib.load(model_path)
    st.write("Model loaded successfully!")
except Exception as e:
    # Display a detailed error message
    st.error("An error occurred while loading the model.")
    st.write("### Exception Details:")
    st.write(f"- **Type:** {type(e).__name__}")
    st.write(f"- **Message:** {str(e)}")
    st.write("### Full Traceback:")
    st.text(traceback.format_exc())
