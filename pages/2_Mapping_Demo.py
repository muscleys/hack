# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from urllib.error import URLError

import streamlit as st
import pydeck as pdk

# List to store points
points = []

def mapping_demo():
    # Input fields for latitude and longitude
    latitude = st.number_input("Enter Latitude:")
    longitude = st.number_input("Enter Longitude:")

    # Slider to control the zoom level
    zoom_level = st.slider("Zoom Level", min_value=1, max_value=20, value=1)

    # Create or get the session state
    session_state = st.session_state
    if 'points' not in session_state:
        session_state.points = []
    
    if 'current_point_index' not in session_state:
        session_state.current_point_index = 0

    if zoom_level < 10:
        scale = zoom_level ** 2
    else:
        scale = zoom_level ** 3

    try:
        # Button to add the point
        show_button = st.sidebar.button("Show Point")

        # Button to delete all points
        reset_button = st.sidebar.button("Delete Points")

        # Button to navigate to the previous point
        previous_button = st.sidebar.button("Previous Point")
        # Button to navigate to the next point
        next_button = st.sidebar.button("Next Point")

        if show_button:
            # Add the current point to the list
            session_state.points.append({"lon": longitude, "lat": latitude})

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
                "HexagonLayer",
                data=session_state.points,
                get_position=["lon", "lat"],
                color=[255, 255, 255, 0],  # Red color with 50% transparency
                radius=400000 / scale,  # Adjust the radius based on the selected zoom level
                elevation_scale=4,
                elevation_range=[0, 1000],
                extruded=True,
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
                        "pitch": 50,
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
                        "pitch": 50,
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

st.set_page_config(page_title="Mapping Demo", page_icon="🌍")
st.markdown("# Points on the map")
st.sidebar.header("Point add and navigation")
st.write(
    """This page gives the opportunity to put points according to the coordinates and navigate between them."""
)

mapping_demo()
