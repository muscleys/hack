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

import pandas as pd
import pydeck as pdk

import streamlit as st
from streamlit.hello.utils import show_code


def mapping_demo():
    file_path = '/workspaces/hack/metadata.csv'
    df = pd.read_csv(file_path)
    
    # Filter DataFrame based on the condition where 'plume' is True
    #plume_df = [i for i in df if i[2] == "yes"]
    plume_df = df.loc[df["plume"]=="yes"]
    try:

         # Slider to control the zoom level
        zoom_level = st.slider("Zoom Level", min_value=1, max_value=20, value=1)

        if zoom_level < 10:
            scale = zoom_level ** 2
        else:
            scale = zoom_level ** 3

        Col1, Col2 = st.columns([2,2])

        # Button to navigate to the previous point
        with Col1 : 
            previous_button = st.button("Previous Point")
        # Button to navigate to the next point
        with Col2 : 
            next_button = st.button("Next Point")



        ALL_LAYERS = {
            "Point": pdk.Layer(
                 "ScatterplotLayer",
                data=plume_df,
                get_position=['lon', 'lat'],
                get_color=[255, 0, 0],  # Red color
                get_radius=400000/scale,  # Adjust the radius based on the desired circle size
                pickable=True,
            ),
            
        }
        
        session_state = st.session_state

        if previous_button:
            # Navigate to the previous point
            former_point = session_state.current_point_index
            session_state.current_point_index = max(0, session_state.current_point_index - 1)
            if former_point == session_state.current_point_index : 
                session_state.current_point_index = len(plume_df) - 1
        
        if next_button:
            # Navigate to the next point
            former_point = session_state.current_point_index
            session_state.current_point_index = min(len(plume_df) - 1, session_state.current_point_index + 1)
            if former_point == session_state.current_point_index : 
                session_state.current_point_index = 0      
        
        if 'current_point_index' not in session_state:
            session_state.current_point_index = 0

        st.sidebar.markdown("### Map Layers")
        selected_layers = [
            layer
            for layer_name, layer in ALL_LAYERS.items()
            if st.sidebar.checkbox(layer_name, True)
        ]
        if selected_layers:
            st.pydeck_chart(
                pdk.Deck(
                    map_style=None,
                    initial_view_state={
                        "latitude": plume_df["lat"][session_state.current_point_index],
                        "longitude": plume_df["lon"][session_state.current_point_index],
                        "zoom": zoom_level,
                        "pitch": 0,
                    },
                    layers=selected_layers,
                )
            )
        
        else:
            st.error("Please choose at least one layer above.")
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

