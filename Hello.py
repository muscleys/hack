import io
from reportlab.pdfgen import canvas
import streamlit as st
from streamlit.logger import get_logger
from PIL import Image
import os
from reportlab.lib.utils import ImageReader
import tifffile as tiff
import numpy as np 
from urllib.error import URLError
import pydeck as pdk
import datetime
import requests
from PIL import ImageDraw
import torch 
from torchvision import transforms
import torch.nn as nn 
import folium

class SimpleCNN(nn.Module):
    def _init_(self, num_classes=2, dropout_prob=0.5):
        super(SimpleCNN, self)._init_()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)  # Dropout layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)  # Dropout layer
        self.fc1 = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        # Explicitly handle the number of input channels
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add a channel dimension
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.dropout1(x)  # Apply dropout
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.dropout2(x)  # Apply dropout
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        return x



#loading the model 
loaded_model = SimpleCNN()
loaded_model = torch.load("./good_model.pth")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])





#tained model
def model(image) : 
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = loaded_model(input_tensor)
    
    # Convert output to probabilities using softmax
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the predicted class index
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class,[20,20] #à modifier



def create_pdf(img_reader, result, selected_date, longitude, latitude, session_state):
    # Create a byte stream buffer
    pdf_buffer = io.BytesIO()

    # Create a canvas to draw on the PDF
    c = canvas.Canvas(pdf_buffer)
    
    if result == 1 : 
        plume = "yes"
    else : 
        plume = "no"

    # Draw something on the PDF
    c.drawString(100, 800, "There is a plume : " + plume)
    c.drawString(100, 700, f"Selected Date: {selected_date}")
    c.drawString(100, 600, f"Selected longitude: {longitude}")
    c.drawString(100, 500, f"Selected latitude: {latitude}")
    c.drawImage(img_reader, x=400, y=600)

    
    

    # Add the point to the session state
    
    point_color = [255, (1-result)*250, 0, 255]  # Red color
    session_state.points.append({"lat": latitude, "lon": longitude, "col": point_color})

    # Finalize the PDF file
    c.showPage()
    c.save()

    # Move the buffer cursor to the start
    pdf_buffer.seek(0)

    return pdf_buffer

def main():
    
    st.title("CleanR Methane Detection")
    st.text("To detect if your satellite footage shows a plume, upload an image. You can export the result as a PDF file.")

    selected_date = st.date_input("Select a date", datetime.date.today())
    # Input fields for latitude and longitude
    col1, col2 = st.columns(2) 
    with col1 : 
        latitude = st.number_input("Latitude", key="latitude")
    with col2 : 
        longitude = st.number_input("Longitude", key="longitude")

    # File upload widget
    uploaded_file = st.file_uploader("Upload a file", type=["tif", "tiff", "pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded file
        tiff_data = tiff.imread(uploaded_file)
        normalized_data = (tiff_data - np.min(tiff_data)) / (np.max(tiff_data) - np.min(tiff_data)) * 255
        normalized_data = normalized_data.astype(np.uint8)
        normalized_data = np.array(normalized_data)

        # Display image if the uploaded file is an image
        st.image(normalized_data, caption="Uploaded Image", use_column_width=True)       
        
        # Creates the display with a square on the plume for the PDF
        image =  Image.fromarray(normalized_data)
        
        # Apply the model on the image 
        result, coordinates = model(image)
        st.write(f"This is the predicted result : {result}")

        draw = ImageDraw.Draw(image)
        draw.rectangle((coordinates[0]-5,coordinates[1]-5,coordinates[0]+5,coordinates[1]+5), fill="white")
        
        img_reader = ImageReader(image)
    
    session_state = st.session_state
    if 'points' not in session_state:
        session_state.points = []
    

    if 'current_point_index' not in session_state:
        session_state.current_point_index = 0


    # Button to generate and download the PDF
    if st.button("Generate PDF"):
        

        pdf_file = create_pdf(img_reader, result, selected_date, latitude, longitude, session_state)

        # Download button
        st.download_button(
            label="Download PDF",
            data=pdf_file,
            file_name="example.pdf",
            mime="application/pdf"
        )

        # Update the map to display the new point
        

    st.markdown("# See where the release areas are on the map")
    st.write(
    """This part gives the opportunity to see points from analyzed pictures : if red then plume else no plume."""
)
    # Slider to control the zoom level
    zoom_level = st.slider("Zoom Level", min_value=1, max_value=15, value=1)

    # Create or get the session state
    
    if zoom_level < 10:
        scale = zoom_level ** 2
    else:
        scale = zoom_level ** 3

    try:
        
        Col1, Col2, Col3 = st.columns([2,2,2])
        
        # Button to delete all points
        with Col1 : 
            reset_button = st.button("Delete Points")

        # Button to navigate to the previous point
        with Col2 : 
            previous_button = st.button("Previous Point")
        # Button to navigate to the next point
        with Col3 : 
            next_button = st.button("Next Point")

        
        if reset_button:
            # Delete all points
            session_state.points = []
            session_state.current_point_index = 0

        if previous_button:
            # Navigate to the previous point
            former_point = session_state.current_point_index
            session_state.current_point_index = max(0, session_state.current_point_index - 1)
            if former_point == session_state.current_point_index : 
                session_state.current_point_index = len(session_state.points) - 1
        
        if next_button:
            # Navigate to the next point
            former_point = session_state.current_point_index
            session_state.current_point_index = min(len(session_state.points) - 1, session_state.current_point_index + 1)
            if former_point == session_state.current_point_index : 
                session_state.current_point_index = 0      

        ALL_LAYERS = {
            "Point": pdk.Layer(
                 "ScatterplotLayer",
                data=session_state.points,
                get_position=['lon', 'lat'],
                get_color='col',  
                get_radius=400000/scale,  # Adjust the radius based on the desired circle size
                pickable=True,
            ),
            
        }

        if len(session_state.points)>=1 : 
            # Include the layer in the Deck object
            st.pydeck_chart(
                pdk.Deck(
                    map_style=None,
                    initial_view_state={
                        "latitude": session_state.points[session_state.current_point_index]["lat"],
                        "longitude": session_state.points[session_state.current_point_index]["lon"],
                        "zoom": zoom_level,
                        "pitch": 0,
                    },
                    layers=list(ALL_LAYERS.values()),  # Add this line to include the layer
                )
            )

        else : 
            st.pydeck_chart(
                pdk.Deck(
                    map_style=None,
                    initial_view_state={
                        "latitude": 0,
                        "longitude": 0,
                        "zoom": zoom_level,
                        "pitch": 0,
                    },
                    layers=list(ALL_LAYERS.values()),  # Add this line to include the layer
                )
            )

    except URLError as e:
        st.error(
            """
            **This demo requires internet access.**
            Connection error: %s
        """
        % e.reason
    )





if __name__ == "__main__":
    main()




def google_search(api_key, cse_id, query):
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={query} methane"
    response = requests.get(url)
    data = response.json()
    return data

# Streamlit app
st.title("Search about Methan release in a suspected region")

# API key and CSE ID
api_key = "AIzaSyDFeh5rkItzfBqns5dtvAk_BTmqOqXyN6c"
cse_id = "905ecee9b8b59471b"

# Input field for country search
country_query = st.text_input("Enter a region to get the most important pages related to methan release in this region:") + "methan"

# Button to trigger the search
search_button = st.button("Search")

if search_button and country_query:
    # Perform Google search
    search_results = google_search(api_key, cse_id, country_query)

    # Display the search results
    if "items" in search_results:
        st.header("Search Results:")
        for item in search_results["items"]:
            st.write(f"- {item['title']}")
            st.write(f"  {item['link']}")
    else:
        st.warning("No search results found.")

