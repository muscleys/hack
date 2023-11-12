import io
from reportlab.pdfgen import canvas
import streamlit as st
from streamlit.logger import get_logger
from PIL import Image
import os
from reportlab.lib.utils import ImageReader
import tifffile as tiff
import numpy as np 


def create_pdf(img_reader):
    # Create a byte stream buffer
    pdf_buffer = io.BytesIO()

    # Create a canvas to draw on the PDF
    c = canvas.Canvas(pdf_buffer)

    # Draw something on the PDF
    c.drawString(100, 750, "Hello, this is a PDF file.")
    c.drawImage(img_reader, x=0, y=500)


    # Finalize the PDF file
    c.showPage()
    c.save()

    # Move the buffer cursor to the start
    pdf_buffer.seek(0)

    return pdf_buffer

def dispose_file(file):
    # Your disposal logic goes here
    # In this example, we simply delete the file
    file_path = os.path.join("uploads", file.name)
    os.remove(file_path)
    st.success(f"File '{file.name}' has been disposed of.")



def main():
    st.title("PDF Generation Example")

    # File upload widget
    uploaded_file = st.file_uploader("Upload a file", type=["tif", "tiff", "pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded file
        st.success("File successfully uploaded!")
        st.write("File name:", uploaded_file.name)
        st.write("File type:", uploaded_file.type)
        st.write("File size:", uploaded_file.size, "bytes")
        tiff_data = tiff.imread(uploaded_file)
        normalized_data = (tiff_data - np.min(tiff_data)) / (np.max(tiff_data) - np.min(tiff_data)) * 255
        normalized_data = normalized_data.astype(np.uint8)
        normalized_data = np.array(normalized_data)
        st.image(normalized_data, caption="Uploaded Image", use_column_width=True)       
        # Display image if the uploaded file is an image
         
        img_reader = ImageReader(Image.fromarray(normalized_data))
    # Button to generate and download the PDF
    if st.button("Generate PDF"):
        pdf_file = create_pdf(img_reader)

        # Download button
        st.download_button(
            label="Download PDF",
            data=pdf_file,
            file_name="example.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()