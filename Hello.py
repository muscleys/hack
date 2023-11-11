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
from streamlit.logger import get_logger
from PIL import Image

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="",
    )

    # File upload widget
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "csv", "pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded file
        st.success("File successfully uploaded!")
        st.write("File name:", uploaded_file.name)
        st.write("File type:", uploaded_file.type)
        st.write("File size:", uploaded_file.size, "bytes")

        # Display image if the uploaded file is an image
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Add a button to dispose of the file
        if st.button("Dispose of File"):
            dispose_file(uploaded_file)

      


def dispose_file(file):
    # Your disposal logic goes here
    # In this example, we simply delete the file
    file_path = os.path.join("uploads", file.name)
    os.remove(file_path)
    st.success(f"File '{file.name}' has been disposed of.")

if __name__ == "__main__":
    run()
