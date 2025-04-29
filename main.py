"""
Docling Streamlit Document Converter App

Author: bgeneto
Date: 2025-04-29
Version: 1.2.3

This Streamlit application provides a user interface for converting documents (PDF, DOCX, PPTX, images, etc.) using the Docling library. It supports OCR, table detection, and exporting to Markdown, plain text, JSON, and HTML. Users can configure conversion options and download results in various formats.
"""

import streamlit as st
import tempfile, os, time, json
from pathlib import Path
from typing import Dict, Any
import io
import zipfile

# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc import ImageRefMode

# ----------------------------------------------------------------------------
# Helpers & Cache
# ----------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def convert_document(
    file_bytes: bytes,
    file_name: str,
    do_ocr: bool,
    do_tables: bool,
    match_cells: bool,
    ocr_dpi: int = 300,
    force_full_page_ocr: bool = False,  # Default to False as per your request
    include_images: bool = True,  # Added parameter for including images
) -> Dict[str, Any]:
    """
    Convert a document using Docling and return multiple export formats.

    Args:
        file_bytes (bytes): The file content as bytes.
        file_name (str): The name of the file.
        do_ocr (bool): Whether to enable OCR.
        do_tables (bool): Whether to detect tables.
        match_cells (bool): Whether to match table cells.
        ocr_dpi (int): OCR DPI setting (not used by TesseractCliOcrOptions).
        force_full_page_ocr (bool): Force full page OCR if True.
        include_images (bool): Whether to include figures/images in output.

    Returns:
        Dict[str, Any]: Dictionary with export results and metadata.
    """
    # STEP 1: configure your PDF pipeline
    pdf_opts = PdfPipelineOptions()
    pdf_opts.do_ocr = do_ocr
    pdf_opts.do_table_structure = do_tables
    pdf_opts.table_structure_options.do_cell_matching = match_cells

    # Enable image extraction based on include_images
    pdf_opts.images_scale = 2.0
    pdf_opts.generate_page_images = include_images
    pdf_opts.generate_picture_images = include_images

    # Configure OCR options if enabled
    if do_ocr:
        ocr_options = TesseractCliOcrOptions(
            lang=["auto"],  # Enable auto language detection
            force_full_page_ocr=force_full_page_ocr,  # Moved to OCR options
        )
        pdf_opts.ocr_options = ocr_options

    # STEP 2: build the converter, specifying the pipeline class & backend
    converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.IMAGE,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.MD,
        ],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=StandardPdfPipeline,
                backend=PyPdfiumDocumentBackend,
                pipeline_options=pdf_opts,
            )
        },
    )

    # STEP 3: dump bytes → temp file so docling can read from disk
    suffix = Path(file_name).suffix
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(file_bytes)
    tf.flush()
    tf.close()

    # STEP 4: convert
    t0 = time.time()
    result = converter.convert(tf.name)
    elapsed = time.time() - t0

    # STEP 5: pull out your exports
    base = Path(file_name).stem

    # Markdown with embedded images
    md_path = Path(tempfile.gettempdir()) / f"{base}-with-images.md"
    result.document.save_as_markdown(md_path, image_mode=ImageRefMode.EMBEDDED)
    with open(md_path, "r", encoding="utf-8") as f:
        m = f.read()
    os.remove(md_path)

    # Plain text without markdown markers
    txt_path = Path(tempfile.gettempdir()) / f"{base}.txt"
    result.document.save_as_markdown(
        txt_path,
        image_mode=ImageRefMode.PLACEHOLDER,
        strict_text=True,
    )
    with open(txt_path, "r", encoding="utf-8") as f:
        t = f.read()
    os.remove(txt_path)

    # HTML with embedded images (self-contained)
    output_dir = tempfile.TemporaryDirectory()
    html_path = Path(output_dir.name) / f"{base}.html"
    result.document.save_as_html(html_path, image_mode=ImageRefMode.EMBEDDED)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.write(html_path, arcname=f"{base}.html")
    zip_buf.seek(0)

    # STEP 6: clean up
    try:
        os.remove(tf.name)
    except OSError:
        pass
    output_dir.cleanup()

    return {
        "base": base,
        "elapsed": elapsed,
        "json": json.dumps(result.document.export_to_dict(), indent=2),
        "text": t,
        "markdown": m,
        "html_zip": zip_buf.read(),
        "ocr_settings": (
            {
                "dpi": ocr_dpi,
                "auto_language": True,
                "force_full_page_ocr": force_full_page_ocr,
            }
            if do_ocr
            else None
        ),
    }


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Doc Converter", layout="wide")
st.title("📄 Docling Document Converter")

# Sidebar for pipeline options
with st.sidebar:
    st.header("..:: OPTIONS ::..")

    # General options
    st.subheader(":gear: General Settings")
    do_tables = st.checkbox("Detect tables", value=True)
    match_cells = st.checkbox("Match table cells", value=True)

    # OCR options
    st.subheader(":pencil2: OCR Settings")
    do_ocr = st.checkbox(
        "Enable OCR",
        value=False,
        help="Enable only if text is missing in the output.",
    )

    if do_ocr:
        # DPI setting
        ocr_dpi = st.slider(
            "OCR DPI",
            min_value=100,
            max_value=600,
            value=300,
            step=50,
            help="Higher DPI improves accuracy but increases processing time.",
        )

        force_full_page_ocr = st.checkbox(
            "Force Full Page OCR",
            value=False,  # Default to False as per your request
            help="Process entire page even if text detection finds text blocks.",
        )

        st.warning("💡 We will use automatic language detection for OCR.")

    # Image/Figure options
    st.subheader(":frame_with_picture: Image Settings")
    include_images = st.checkbox("Include figures/images", value=True)

    # Compression option
    st.subheader(":package: Output Settings")
    compress_output = st.checkbox(
        "Compress/zip all files",
        value=False,
        help="Download all output formats in a single zip file.",
    )

# Add some helpful documentation in an expander
with st.expander("ℹ️ How to use this converter"):
    st.markdown(
        """
        ### Docling Converter Guide

        **1. Upload Files:**
        - Supported formats: PDF, PPTX, DOCX, MD, and images (PNG, JPG, JPEG, GIF, BMP, TIFF)
        - You can upload multiple files at once (and also use drag & drop)

        **2. General Settings:**
        - **Detect tables:** Extract table structures from documents
        - **Match table Cells:** Improve table structure recognition

        **3. OCR Settings:**
        - **Enable OCR:** Optical Character Recognition for scanned documents/images
        - **OCR DPI:** Higher values improve quality but increase processing time
        - **Force Full Page OCR:** Process entire page even if text detection finds text blocks

        **3. Image Settings:**
        - **Include figures/images:** Include images in the output formats that supports it, like Markdown and HTML

        **4. Output Settings:**
        - **Compress/zip all files:** Download all output formats in a single zip file (disable to download each format separately)
        - Get your documents in JSON, Text, Markdown, or HTML format

        **Tips for Best Results:**
        - You can preview each format before downloading (by clicking in the "👁️ Preview Content" area)
        - For scanned documents without embeded text, use high DPI (300-400)
        - For documents with tables, enable both table detection options
        - Click "Convert again!" when changing settings for consistent results
        """
    )

# File uploader
uploaded = st.file_uploader(
    "Pick one or more files (pdf, pptx, docx, md, images)...",
    type=["pdf", "pptx", "docx", "md", "png", "jpg", "jpeg", "gif", "bmp", "tiff"],
    accept_multiple_files=True,
)

# Make sure we have a session‐scope holder for our conversions
if "conversions" not in st.session_state:
    st.session_state.conversions = {}

# As soon as the user picks files, loop & cache‐convert each
if uploaded:
    for up in uploaded:
        key = up.name
        if key not in st.session_state.conversions:
            # Show spinner while converting
            with st.spinner(f"Please wait, converting {up.name}..."):
                data = convert_document(
                    up.read(),
                    up.name,
                    do_ocr,
                    do_tables,
                    match_cells,
                    ocr_dpi if do_ocr else 300,
                    force_full_page_ocr if do_ocr else False,
                    include_images=include_images,
                )
                st.session_state.conversions[key] = data
            st.success(f"Finished converting {up.name}")

# Now render all the results (if any) with persistent download buttons
if compress_output and st.session_state.conversions:
    # Create a zip with all formats for all files
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for fname, data in st.session_state.conversions.items():
            zf.writestr(f"{data['base']}.md", data["markdown"])
            zf.writestr(f"{data['base']}.txt", data["text"])
            zf.writestr(f"{data['base']}.json", data["json"])
            # Extract HTML from the existing html_zip
            import zipfile as zfmod
            import io as iomod

            with zfmod.ZipFile(iomod.BytesIO(data["html_zip"])) as htmlzip:
                for name in htmlzip.namelist():
                    if name.endswith(".html"):
                        with htmlzip.open(name) as html_file:
                            html_content = html_file.read()
                        zf.writestr(f"{data['base']}.html", html_content)
                        break
    zip_buf.seek(0)
    st.download_button(
        "⬇️ Download All Converted Files (zip)",
        zip_buf,
        file_name="all_converted_files.zip",
        mime="application/zip",
    )
    # Optionally, still show previews for each file
    for fname, data in st.session_state.conversions.items():
        st.markdown(f"---\n### {fname}")
        st.write(f"⏱ Converted in **{data['elapsed']:.2f}** seconds")
        if data["ocr_settings"]:
            settings = data["ocr_settings"]
            st.caption(
                f"OCR Settings: DPI={settings['dpi']}, "
                f"Auto Language Detection, "
                f"Full Page OCR={'Yes' if settings['force_full_page_ocr'] else 'No'}"
            )
        with st.expander("👁️ Preview Content"):
            tab1, tab2, tab3, tab4 = st.tabs(["Markdown", "Text", "JSON", "HTML"])
            with tab1:
                st.markdown(data["markdown"])
            with tab2:
                st.text(data["text"])
            with tab3:
                st.json(data["json"])
            with tab4:
                import zipfile
                import io

                html_content = None
                try:
                    with zipfile.ZipFile(io.BytesIO(data["html_zip"])) as zf:
                        for name in zf.namelist():
                            if name.endswith(".html"):
                                with zf.open(name) as html_file:
                                    html_content = html_file.read().decode(
                                        "utf-8", errors="replace"
                                    )
                                break
                    if html_content:
                        st.components.v1.html(html_content, height=400, scrolling=True)
                    else:
                        st.warning("No HTML file found in archive for preview.")
                except Exception as e:
                    st.error(f"Error extracting HTML preview: {e}")
else:
    for fname, data in st.session_state.conversions.items():
        st.markdown(f"---\n### {fname}")
        st.write(f"⏱ Converted in **{data['elapsed']:.2f}** seconds")
        if data["ocr_settings"]:
            settings = data["ocr_settings"]
            st.caption(
                f"OCR Settings: DPI={settings['dpi']}, "
                f"Auto Language Detection, "
                f"Full Page OCR={'Yes' if settings['force_full_page_ocr'] else 'No'}"
            )
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.download_button(
            "⬇️ Download Markdown",
            data["markdown"],
            file_name=f"{data['base']}.md",
            mime="text/markdown",
        )
        col2.download_button(
            "⬇️ Download Text",
            data["text"],
            file_name=f"{data['base']}.txt",
            mime="text/plain",
        )
        col3.download_button(
            "⬇️ Download JSON",
            data["json"],
            file_name=f"{data['base']}.json",
            mime="application/json",
        )
        col4.download_button(
            "⬇️ Download HTML",
            data["html_zip"],
            file_name=f"{data['base']}.zip",
            mime="application/zip",
        )
        with st.expander("👁️ Preview Content"):
            tab1, tab2, tab3, tab4 = st.tabs(["Markdown", "Text", "JSON", "HTML"])
            with tab1:
                st.markdown(data["markdown"])
            with tab2:
                st.text(data["text"])
            with tab3:
                st.json(data["json"])
            with tab4:
                import zipfile
                import io

                html_content = None
                try:
                    with zipfile.ZipFile(io.BytesIO(data["html_zip"])) as zf:
                        for name in zf.namelist():
                            if name.endswith(".html"):
                                with zf.open(name) as html_file:
                                    html_content = html_file.read().decode(
                                        "utf-8", errors="replace"
                                    )
                                break
                    if html_content:
                        st.components.v1.html(html_content, height=400, scrolling=True)
                    else:
                        st.warning("No HTML file found in archive for preview.")
                except Exception as e:
                    st.error(f"Error extracting HTML preview: {e}")

# Add some statistics if files have been processed
if st.session_state.conversions:
    st.markdown("---")
    st.subheader("Conversion Statistics")

    total_files = len(st.session_state.conversions)
    total_time = sum(data["elapsed"] for data in st.session_state.conversions.values())
    avg_time = total_time / total_files if total_files > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Files Processed", total_files)
    col2.metric("Total Processing Time", f"{total_time:.2f} seconds")
    col3.metric("Average Time per File", f"{avg_time:.2f} seconds")

    # Add a clear cache button
    if st.button("Convert again!"):
        st.session_state.conversions = {}
        st.rerun()

# Add footer
current_year = time.strftime("%Y")
st.markdown("---")
st.caption(
    f"Docling Document Converter v1.2.3 | Copyright © 2025-{current_year} by bgeneto"
)
