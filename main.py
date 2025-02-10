import streamlit as st
import fitz # PyMuPDF
# from PIL import Image, ImageOps
import cv2
import numpy as np
import io
import time
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from streamlit_pdf_viewer import pdf_viewer

def process_pdf(input_pdf_bytes, dpi, page_callback):
	pdf_document = fitz.open(stream=input_pdf_bytes, filetype="pdf")

	output_buffer = io.BytesIO()
	c = canvas.Canvas(output_buffer)

	num_pages = len(pdf_document)
	page_callback(total=num_pages)

	# todo: multiprocessing? https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html
	for page_number in range(num_pages):
		page = pdf_document.load_page(page_number)
		page_callback(page=True)
		page_width, page_height = page.rect.width, page.rect.height

		# zoom = 2
		# mat = fitz.Matrix(zoom, zoom)
		# pix = page.get_pixmap(matrix=mat)
		pix = page.get_pixmap(dpi=dpi)
		image = np.frombuffer(pix.samples, dtype=np.uint8)
		# image = image.reshape((pix.height, pix.width, 3))

		image = cv2.bitwise_not(image)

		# image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

		# image[:, :, 1] = 255 - image[:, :, 1]  # Invert Saturation
		# image[:, :, 2] = 255 - image[:, :, 2]  # Invert Value
		# image[:, :, 2] = np.clip(image[:, :, 2], 0, 255)

		# image[:, :, 0] = (image[:, :, 0] + 90) % 180

		# image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

		# https://stackoverflow.com/a/65355529
		hue_rotate = np.array([
			[-0.574,  1.43,  0.144],
			[0.426,  0.43,  0.144],
			[0.426,  1.43, -0.856]
		])
		# saturate = np.array([
		# 	[7.87,  -7.15, -0.72],
		# 	[-2.13,   2.85, -0.72],
		# 	[-2.13,  -7.15,  9.28]
		# ])

		image = image.reshape(-1, 3) # (H*W, 3)
		image = image @ hue_rotate.T
		image = np.clip(image, 0, 255)
		image = image.astype(np.uint8).reshape((pix.height, pix.width, 3))

		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		_, buffer = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
		image_bytes = io.BytesIO(buffer.tobytes())

		# A4 size: 595 x 842 pt
		MAX_WIDTH = 792 # 11"
		if page_width > MAX_WIDTH:
			page_height = MAX_WIDTH * page_height / page_width
			page_width = MAX_WIDTH

		c.setPageSize((page_width, page_height))
		img_reader = ImageReader(image_bytes)
		c.drawImage(img_reader, 0, 0, width=page_width, height=page_height)
		c.showPage()

	c.save()
	output_buffer.seek(0)

	return output_buffer

st.set_page_config(page_title="PDF Inverter 🌗", page_icon="🌗")
st.markdown('<div style="text-align: center;font-size:300%;margin-bottom: 30px"><b>PDF Inverter 🌗</b></div>', unsafe_allow_html=True)

def start_processing():
	st.session_state.start_processing = True
	st.session_state.processed_files = []

if 'processed_files' not in st.session_state:
	st.session_state.processed_files = []

if 'start_processing' not in st.session_state:
	st.session_state.start_processing = False

st.markdown("<hr>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
dpi = st.slider("Output Quality (DPI)", 100, 800, 150, step=50)
st.markdown("<hr>", unsafe_allow_html=True)

if uploaded_files:
	st.markdown('<div style="text-align: center;font-size:170%;margin-bottom: 10px"><b>🛠 Process PDFs</b></div>', unsafe_allow_html=True)

	if st.button("🖨️ START 🖨️", use_container_width=True, key="start_button"):
		start_processing()
		total_files = len(uploaded_files)
		total_pages = 0 # updated
		completed_pages = 0
		progress_text = st.empty()
		file_counter = st.empty()
		total_progress = st.progress(0)

		# todo: make this prettier, add subprogress (extracting, inverting, etc.)
		def page_callback(page=None, total=None):
			global total_pages, completed_pages
			if total is not None:
				total_pages += total
			if page is not None:
				completed_pages += 1
			total_progress.progress(completed_pages / total_pages)

		for i, uploaded_file in enumerate(uploaded_files):
			file_counter.markdown(
				f"<div style='text-align: center; font-size: 24px; margin-bottom: 20px;'>"
				f"<b>Processing file {i+1} of {total_files}</b><br>",
				unsafe_allow_html=True
			)
			with st.spinner(f"Processing {uploaded_file.name}"):
				binary_pdf = uploaded_file.read()
				input_pdf_stream = io.BytesIO(binary_pdf)
				input_pdf_stream.seek(0)

				output_pdf_stream = process_pdf(input_pdf_stream.getvalue(), dpi, page_callback)

				output_stream = output_pdf_stream.getvalue()
				st.session_state.processed_files.append(
					(uploaded_file.name, output_stream, f"download_{i}_{time.time()}")
				)

			# total_progress.progress((i + 1) / total_files)

		st.markdown("<hr>", unsafe_allow_html=True)
		st.markdown('<div style="text-align: center;font-size:170%;margin-bottom: 10px"><b>🔎 View Processed PDFs</b></div>', unsafe_allow_html=True)

	for j, (file_name, output_stream, unique_key) in enumerate(st.session_state.processed_files):
		st.markdown("<hr>", unsafe_allow_html=True)
		st.markdown(f'<div style="text-align: center;font-size:170%;margin-bottom: 10px"><b>🔎 View PDF</b></div>', unsafe_allow_html=True)
		col1, col2 = st.columns([3, 1])
		with col1:
			st.write(f"➔ {file_name} ✅ ({len(output_stream) / 1024:,.2f} KB)")
		with col2:
			st.download_button(
				label="Download",
				data=output_stream,
				file_name=f"{file_name.removesuffix('.pdf')}_i.pdf",
				mime="application/pdf",
				use_container_width=True,
				key=f"download_{j}_{time.time()}"
			)

		# Display the output PDF using streamlit_pdf_viewer
		pdf_viewer(
			output_stream,
			width=1200,
			height=600,
			pages_vertical_spacing=2,
			annotation_outline_size=2,
			pages_to_render=[]
		)