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


def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
    session = requests.Session()

    # Display starting message
    st.write("Initiating download from Google Drive...")
    
    # Initial request
    response = session.get(url, stream=True)
    st.write("Checking for download confirmation cookies...")

    # Handle confirmation cookies for large files
    for key, value in response.cookies.items():
        st.write(f"Cookie: {key} = {value}")
        if key.startswith("download_warning"):
            st.write("Large file detected. Handling Google Drive confirmation step...")
            url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"

    # Start downloading the file
    response = session.get(url, stream=True)
    
    # Progress bar for the download
    progress = st.progress(0)
    downloaded_size = 0

    # Save file in chunks
    with open(destination, "wb") as file:
        for chunk in response.iter_content(32768):  # Chunk size = 32 KB
            if chunk:  # Filter out keep-alive chunks
                file.write(chunk)
                downloaded_size += len(chunk)
                progress.progress(downloaded_size / 100000000)

    st.success(f"3 Download complete! File saved as {destination}")

    file_size = os.path.getsize(destination)
    st.write(f"3 File size: {file_size} bytes")

    # Display the first 10 characters
    with open(destination, "rb") as file:
        first_10_chars = file.read(200)
        st.write(f"First 10 characters: {first_10_chars}")

    return destination

# Example usage
destination = "reg_rf.pkl"

try:
    # Download and handle the file
    model_path = download_file_from_google_drive(file_id, destination)
    st.write("Loading the model...")
    import joblib
    model = joblib.load(model_path)
    st.success("Model loaded successfully and ready for predictions!")
except Exception as e:
    st.error("An error occurred during file download or model loading.")
    st.write("### Exception Details:")
    st.write(f"- **Type:** {type(e).__name__}")
    st.write(f"- **Message:** {str(e)}")
    import traceback
    st.write("### Full Traceback:")
    st.text(traceback.format_exc())








