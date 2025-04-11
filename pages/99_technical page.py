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



file_id='1i6dUP4QvaAHP3W-wxLc2A9WtpISuU4RU' 


def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    session = requests.Session()

    # Display starting message
    st.write("Initiating download from Google Drive...")
    
    # Initial request
    response = session.get(url, stream=True)
    st.write("Checking for download confirmation cookies...")

    # Handle confirmation cookies for large files
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            st.write("Large file detected. Handling Google Drive confirmation step...")
            url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={value}"

    # Start downloading the file
    response = session.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    st.write(f"Downloading file: {destination} (Size: {total_size / (1024 * 1024):.2f} MB)")
    
    # Progress bar for the download
    progress = st.progress(0)
    downloaded_size = 0

    # Save file in chunks
    with open(destination, "wb") as file:
        for chunk in response.iter_content(32768):  # Chunk size = 32 KB
            if chunk:  # Filter out keep-alive chunks
                file.write(chunk)
                downloaded_size += len(chunk)
                progress.progress(downloaded_size / total_size)

    st.success(f"Download complete! File saved as {destination}")
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








