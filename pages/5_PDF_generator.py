import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

def analyze_image(image_path):
    # Your image analysis logic goes here
    # For demonstration purposes, let's assume the result is a string
    result = "Analysis Result: This is an example analysis result."
    return result

def generate_pdf(image_path, result):
    pdf_filename = "analysis_report.pdf"
    img = Image.open(image_path)

    # Create PDF
    c = canvas.Canvas(pdf_filename, pagesize=letter)

    # Draw image on the PDF using ImageReader
    img_reader = ImageReader(image_path)
    c.drawImage(img_reader, 100, 500, width=img.width, height=img.height)

    # Add result text to the PDF
    c.setFont("Helvetica", 12)
    c.drawString(100, 450, result)

    # Save the PDF
    c.save()

    return pdf_filename

# Streamlit app
st.title("Image Analysis Report")

# File uploader for image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Button to analyze and generate PDF
    if st.button("Analyze and Generate PDF"):
        # Perform image analysis
        result = analyze_image(uploaded_file)

        # Generate PDF
        pdf_filename = generate_pdf(uploaded_file, result)

        # Offer the PDF file for download
        st.markdown(f"### [Download PDF]({pdf_filename})")
