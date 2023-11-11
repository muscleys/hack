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

def mapping_demo():
    # Input fields for latitude and longitude
    latitude = st.sidebar.number_input("Enter Latitude:")
    longitude = st.sidebar.number_input("Enter Longitude:")
    
    # Button to add the point
    show_button = st.sidebar.button("Show Point")
    #button to delete all points 
    reset_button = st.sidebar.button("Delete Points")

    # List to store points
    points = []

    try:
        if show_button:
            # Add the current point to the list
            points.append({"lon": longitude, "lat": latitude})

        if reset_button:
            #deletes all points 
            points = []

        ALL_LAYERS = {
            "Point": pdk.Layer(
                "HexagonLayer",
                data=points,
                get_position=["lon", "lat"],
                color=[255, 255, 255, 0],  # Red color with 50% transparency
                radius=400000,
                elevation_scale=4,
                elevation_range=[0, 1000],
                extruded=True,
            ),
        }
        
        # Include the layer in the Deck object
        st.pydeck_chart(
            pdk.Deck(
                map_style=None,
                initial_view_state={
                    "latitude": 0,
                    "longitude": 0,
                    "zoom": 1,
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
st.markdown("# Mapping Demo")
st.sidebar.header("Mapping Demo")
st.write(
    """This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/library/api-reference/charts/st.pydeck_chart)
to display geospatial data."""
)

mapping_demo()
