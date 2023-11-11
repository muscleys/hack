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


import streamlit as st
import requests

def google_search(api_key, cse_id, query):
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={query} methane"
    response = requests.get(url)
    data = response.json()
    return data

# Streamlit app
st.title("Google Search about Country and Methane")

# API key and CSE ID
api_key = "AIzaSyDFeh5rkItzfBqns5dtvAk_BTmqOqXyN6c"
cse_id = "905ecee9b8b59471b"

# Input field for country search
country_query = st.text_input("Enter a country:") + "methan"

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

