"""
Docling Streamlit Document Converter App

Author: bgeneto
Date: 2025-07-08
Version: 1.6.0

This Streamlit application provides a user interface for converting documents (PDF, DOCX, PPTX, XLSX, images, etc.) using the Docling library. It supports both Standard pipeline (OCR, table detection, code/formula enrichment) and VLM pipeline (Vision-Language Model with SmolDocling) for enhanced document understanding. Features robust error handling, automatic accelerator detection, intelligent file validation, progress tracking, and enhanced resource management. Users can configure conversion options and download results in various formats with intelligent preview truncation for large documents.
"""

import streamlit as st
import tempfile, os, time, json
from pathlib import Path
from typing import Dict, Any
import io
import zipfile
import hashlib
import logging
from contextlib import contextmanager

# Docling imports
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
    VlmPipelineOptions,
)
from docling.datamodel import vlm_model_specs
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling_core.types.doc import ImageRefMode
from docling.datamodel.settings import settings

settings.perf.elements_batch_size = 1

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------


@contextmanager
def temporary_file(file_bytes: bytes, suffix: str):
    """Context manager for temporary file handling."""
    tf = None
    try:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tf.write(file_bytes)
        tf.flush()
        tf.close()
        yield tf.name
    finally:
        if tf and os.path.exists(tf.name):
            try:
                os.remove(tf.name)
            except OSError as e:
                logger.warning(f"Failed to remove temporary file {tf.name}: {e}")


def detect_accelerator() -> AcceleratorOptions:
    """Detect available accelerator and configure accordingly."""
    try:
        import torch

        if torch.cuda.is_available():
            logger.info("CUDA detected, using GPU acceleration")
            return AcceleratorOptions(num_threads=8, device=AcceleratorDevice.CUDA)
    except ImportError:
        logger.info("PyTorch not available, falling back to CPU")

    # Fallback to CPU
    logger.info("Using CPU acceleration")
    return AcceleratorOptions(num_threads=8, device=AcceleratorDevice.CPU)


def generate_file_key(file_name: str, file_bytes: bytes) -> str:
    """Generate unique key for file based on name and content hash."""
    content_hash = hashlib.md5(file_bytes).hexdigest()[:8]
    return f"{file_name}_{content_hash}"


def validate_file_size(file_name: str, file_bytes: bytes) -> bool:
    """Validate file size and show appropriate messages."""
    file_size = len(file_bytes)
    if file_size > MAX_FILE_SIZE:
        st.error(
            f"❌ File {file_name} exceeds maximum size of {MAX_FILE_SIZE // 1024 // 1024}MB (current: {file_size // 1024 // 1024}MB)"
        )
        return False
    return True


def validate_configuration(use_vlm_pipeline: bool, do_ocr: bool) -> bool:
    """Validate configuration settings."""
    if use_vlm_pipeline and do_ocr:
        st.warning("⚠️ OCR is automatically handled by VLM pipeline")
    return True


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
    do_code: bool,
    do_formulas: bool,
    ocr_dpi: int = 300,
    force_full_page_ocr: bool = False,
    include_images: bool = True,
    use_vlm_pipeline: bool = False,
    # Cache key parameters - these ensure cache invalidation when settings change
    _cache_buster: str = None,  # Will be populated with all settings as a string
) -> Dict[str, Any]:
    """
    Convert a document using Docling and return multiple export formats.
    Supports both Standard PDF Pipeline and VLM Pipeline with SmolDocling.
    """
    try:
        # Detect available accelerator
        accelerator_options = detect_accelerator()

        if use_vlm_pipeline:
            # Configure VLM pipeline
            vlm_opts = VlmPipelineOptions()
            vlm_opts.accelerator_options = accelerator_options
            vlm_opts.generate_page_images = True
            vlm_opts.vlm_options = vlm_model_specs.SMOLDOCLING_TRANSFORMERS

            converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.IMAGE,
                    InputFormat.DOCX,
                    InputFormat.HTML,
                    InputFormat.PPTX,
                    InputFormat.XLSX,
                    InputFormat.ASCIIDOC,
                    InputFormat.MD,
                ],
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=vlm_opts,
                    ),
                    InputFormat.IMAGE: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=vlm_opts,
                    ),
                },
            )
        else:
            # Configure standard PDF pipeline
            pdf_opts = PdfPipelineOptions()
            pdf_opts.accelerator_options = accelerator_options
            pdf_opts.do_ocr = do_ocr
            pdf_opts.do_table_structure = do_tables
            pdf_opts.table_structure_options.do_cell_matching = match_cells
            pdf_opts.do_code_enrichment = do_code
            pdf_opts.do_formula_enrichment = do_formulas
            pdf_opts.images_scale = 2.0
            pdf_opts.generate_page_images = include_images
            pdf_opts.generate_picture_images = include_images

            if do_ocr:
                ocr_options = TesseractCliOcrOptions(
                    lang=["auto"],
                    force_full_page_ocr=force_full_page_ocr,
                )
                pdf_opts.ocr_options = ocr_options

            converter = DocumentConverter(
                allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.IMAGE,
                    InputFormat.DOCX,
                    InputFormat.HTML,
                    InputFormat.PPTX,
                    InputFormat.XLSX,
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

        # Use context manager for temporary file
        suffix = Path(file_name).suffix
        with temporary_file(file_bytes, suffix) as temp_path:
            # Convert document
            t0 = time.time()
            result = converter.convert(temp_path)
            elapsed = time.time() - t0

        # Process outputs with proper error handling
        base = Path(file_name).stem

        try:
            # Markdown with embedded images (only if images are enabled)
            if include_images:
                md_path = Path(tempfile.gettempdir()) / f"{base}-with-images.md"
                result.document.save_as_markdown(
                    md_path, image_mode=ImageRefMode.EMBEDDED
                )
                with open(md_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
                os.remove(md_path)
            else:
                # Save markdown without any image references
                md_path = Path(tempfile.gettempdir()) / f"{base}-no-images.md"
                result.document.save_as_markdown(
                    md_path, image_mode=ImageRefMode.PLACEHOLDER
                )
                with open(md_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
                os.remove(md_path)

                # Remove image placeholder lines that contain the error message
                lines = markdown_content.split("\n")
                filtered_lines = [
                    line
                    for line in lines
                    if not (
                        line.strip().startswith("<!--")
                        and "Image not available" in line
                        and "generate_picture_images" in line
                    )
                ]
                markdown_content = "\n".join(filtered_lines)

            # Plain text
            txt_path = Path(tempfile.gettempdir()) / f"{base}.txt"
            result.document.save_as_markdown(
                txt_path,
                image_mode=ImageRefMode.PLACEHOLDER,
                strict_text=True,
            )
            with open(txt_path, "r", encoding="utf-8") as f:
                text_content = f.read()
            os.remove(txt_path)

            # HTML with embedded images
            with tempfile.TemporaryDirectory() as output_dir:
                html_path = Path(output_dir) / f"{base}.html"
                result.document.save_as_html(
                    html_path, image_mode=ImageRefMode.EMBEDDED
                )

                zip_buf = io.BytesIO()
                with zipfile.ZipFile(zip_buf, "w") as zf:
                    zf.write(html_path, arcname=f"{base}.html")
                zip_buf.seek(0)
                html_zip_content = zip_buf.read()

        except Exception as e:
            logger.error(f"Error processing outputs for {file_name}: {e}")
            raise

        return {
            "base": base,
            "elapsed": elapsed,
            "json": json.dumps(result.document.export_to_dict(), indent=2),
            "text": text_content,
            "markdown": markdown_content,
            "html_zip": html_zip_content,
            "accelerator": accelerator_options.device.value,
            "ocr_settings": (
                {
                    "dpi": ocr_dpi,
                    "auto_language": True,
                    "force_full_page_ocr": force_full_page_ocr,
                }
                if do_ocr and not use_vlm_pipeline
                else None
            ),
            "vlm_settings": (
                {
                    "pipeline": "VLM with SmolDocling",
                    "model": "SMOLDOCLING_TRANSFORMERS",
                }
                if use_vlm_pipeline
                else None
            ),
        }

    except Exception as e:
        logger.error(f"Conversion failed for {file_name}: {e}")
        st.error(f"❌ Conversion failed for {file_name}: {str(e)}")
        raise


def preview_text_content(
    text: str, max_chars: int = 5000, max_lines: int = 100
) -> tuple[str, bool]:
    """
    Returns a truncated version of text for preview purposes.

    Args:
        text: The full text content
        max_chars: Maximum number of characters to show
        max_lines: Maximum number of lines to show

    Returns:
        tuple: (preview_text, is_truncated)
    """
    lines = text.split("\n")
    is_truncated = False

    # Check if we need to truncate by lines
    if len(lines) > max_lines:
        preview_lines = lines[:max_lines]
        is_truncated = True
    else:
        preview_lines = lines

    preview_text = "\n".join(preview_lines)

    # Check if we need to truncate by characters
    if len(preview_text) > max_chars:
        preview_text = preview_text[:max_chars]
        is_truncated = True

    return preview_text, is_truncated


def preview_json_content(json_str: str, max_chars: int = 5000) -> tuple[str, bool]:
    """
    Returns a truncated version of JSON for preview purposes.
    Tries to keep the JSON structure valid by truncating at reasonable points.

    Args:
        json_str: The full JSON content as string
        max_chars: Maximum number of characters to show

    Returns:
        tuple: (preview_json, is_truncated)
    """
    if len(json_str) <= max_chars:
        return json_str, False

    # Try to find a reasonable truncation point (end of an object or array)
    truncated = json_str[:max_chars]

    # Find the last complete line that ends with }, ], or similar
    lines = truncated.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if line.endswith((",", "}", "]", "}")):
            truncated = "\n".join(lines[: i + 1])
            break

    # Add truncation indicator
    truncated += "\n  ...\n}"

    return truncated, True


def preview_html_content(html_content: str, max_chars: int = 8000) -> tuple[str, bool]:
    """
    Returns a truncated version of HTML for preview purposes.
    Tries to maintain basic HTML structure.

    Args:
        html_content: The full HTML content
        max_chars: Maximum number of characters to show

    Returns:
        tuple: (preview_html, is_truncated)
    """
    if len(html_content) <= max_chars:
        return html_content, False

    # Find a reasonable place to cut off (preferably at the end of a tag)
    truncated = html_content[:max_chars]

    # Try to find the last complete tag
    last_tag_end = truncated.rfind(">")
    if last_tag_end != -1:
        truncated = truncated[: last_tag_end + 1]

    # Add a note about truncation
    truncated += "\n<!-- Content truncated for preview -->"

    return truncated, True


def generate_cache_buster(
    do_ocr: bool,
    do_tables: bool,
    match_cells: bool,
    do_code: bool,
    do_formulas: bool,
    ocr_dpi: int,
    force_full_page_ocr: bool,
    include_images: bool,
    use_vlm_pipeline: bool,
) -> str:
    """
    Generate a cache buster string based on all conversion settings.
    This ensures the cache is invalidated when any setting changes.
    """
    return f"ocr_{do_ocr}_tables_{do_tables}_cells_{match_cells}_code_{do_code}_formulas_{do_formulas}_dpi_{ocr_dpi}_fullpage_{force_full_page_ocr}_images_{include_images}_vlm_{use_vlm_pipeline}"


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Doc Converter", layout="wide", page_icon=":page_facing_up:"
)
st.title("📄 Docling Document Converter")

# Sidebar for pipeline options
with st.sidebar:
    st.header("..:: OPTIONS ::..")

    # Pipeline selection
    st.subheader("🚀 Pipeline Type")
    pipeline_type = st.radio(
        "Choose conversion pipeline:",
        options=["Standard", "VLM"],
        index=0,
        help="Standard: Traditional OCR and table detection. VLM: AI-powered vision-language model (SmolDocling) for better understanding of complex documents.",
    )
    use_vlm_pipeline = pipeline_type == "VLM"

    if use_vlm_pipeline:
        st.info(
            "🤖 VLM Pipeline uses SmolDocling model for enhanced document understanding. Some traditional options may not apply."
        )

    # General options
    st.subheader("⚙️ General Settings")
    do_tables = st.checkbox(
        "Detect tables",
        value=True,
        disabled=use_vlm_pipeline,
        help=(
            "Table detection (disabled for VLM pipeline as it handles structure automatically)"
            if use_vlm_pipeline
            else "Detect table structures in documents"
        ),
    )
    match_cells = st.checkbox(
        "Match table cells",
        value=True,
        disabled=use_vlm_pipeline,
        help=(
            "Table cell matching (disabled for VLM pipeline)"
            if use_vlm_pipeline
            else "Improve table structure recognition"
        ),
    )

    do_code = False
    # Disabled for now, required more vRAM than a 3090 has available
    # do_code = st.checkbox(
    #     "Detect code blocks",
    #     value=False,
    #     disabled=use_vlm_pipeline,
    #     help="Code detection (disabled for VLM pipeline as it handles all content types)" if use_vlm_pipeline else "Identify and format blocks of source code.",
    # )
    do_formulas = False
    # Disabled for now, required more vRAM than a 3090 has available
    # do_formulas = st.checkbox(
    #     "Detect formulas",
    #     value=False,
    #     disabled=use_vlm_pipeline,
    #     help="Formula detection (disabled for VLM pipeline as it handles all content types)" if use_vlm_pipeline else "Identify and format mathematical formulas.",
    # )

    # OCR options
    st.subheader(":pencil2: OCR Settings")
    if use_vlm_pipeline:
        st.info(
            "💡 VLM Pipeline uses vision-language models and doesn't require traditional OCR settings."
        )
        do_ocr = False
        ocr_dpi = 300
        force_full_page_ocr = False
    else:
        do_ocr = st.checkbox(
            "Enable OCR",
            value=False,
            help="Enable only if text is missing in the output.",
        )

        if do_ocr:
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
                value=False,
                help="Process entire page even if text detection finds text blocks.",
            )
            st.warning("💡 We will use automatic language detection for OCR.")
        else:
            ocr_dpi = 300
            force_full_page_ocr = False

    # Image/Figure options
    st.subheader("🖼️ Image Settings")
    include_images = st.checkbox("Include figures/images", value=False)

    # Compression option
    st.subheader(":package: Output Settings")
    compress_output = st.checkbox(
        "Compress/zip all files",
        value=False,
        help="Download all output formats in a single zip file.",
    )

    # Preview settings
    st.subheader("👁️ Preview Settings")
    preview_limit = st.selectbox(
        "Preview size for large documents",
        options=[
            ("Small", 50, 2500),
            ("Medium", 100, 5000),
            ("Large", 200, 10000),
            ("Extra Large", 500, 20000),
        ],
        index=2,  # Default to Large
        format_func=lambda x: x[0],
        help="Choose how much content to show in preview tabs for large documents.",
    )

# Validate configuration after all variables are defined
validate_configuration(use_vlm_pipeline, do_ocr)

# Add some helpful documentation in an expander
with st.expander("ℹ️ How to use this converter"):
    st.markdown(
        """
        ### Docling Converter Guide

        **1. Upload Files:**
        - Supported formats: PDF, DOCX, PPTX, XLSX, HTML, and images (PNG, JPG, JPEG, GIF, BMP, TIFF)
        - You can upload multiple files at once (and also use drag & drop)

        **2. Pipeline Selection:**
        - **Standard Pipeline:** Traditional OCR-based processing with configurable options
        - **VLM Pipeline:** AI-powered Vision-Language Model (SmolDocling) for enhanced document understanding

        **3. General Settings:**
        - **Detect tables:** Extract table structures from documents (Standard pipeline only)
        - **Match table Cells:** Improve table structure recognition (Standard pipeline only)
        - **Detect code blocks:** Identify and format blocks of source code (Standard pipeline only)
        - **Detect formulas:** Identify and format mathematical formulas (Standard pipeline only)

        **4. OCR Settings (Standard Pipeline only):**
        - **Enable OCR:** Optical Character Recognition for scanned documents/images
        - **OCR DPI:** Higher values improve quality but increase processing time
        - **Force Full Page OCR:** Process entire page even if text detection finds text blocks

        **5. Image Settings:**
        - **Include figures/images:** Include images in the output formats that supports it, like Markdown and HTML

        **6. Output Settings:**
        - **Compress/zip all files:** Download all output formats in a single zip file (disable to download each format separately)
        - **Preview size:** Choose how much content to show in preview tabs for large documents
        - Get your documents in JSON, Text, Markdown, or HTML format

        **Tips for Best Results:**
        - **VLM Pipeline:** Best for complex documents with mixed content types, automatically handles tables, formulas, and code
        - **Standard Pipeline:** Faster for simple documents, more configuration options available
        - You can preview each format before downloading (by clicking in the "👁️ Preview Content" area)
        - For large documents, previews are automatically truncated - adjust preview size in settings
        - For scanned documents without embeded text (Standard pipeline), use high DPI (300-400)
        - Click "Convert again!" when changing settings for consistent results
        """
    )

# File uploader
uploaded = st.file_uploader(
    "Pick one or more files (pdf, pptx, docx, xlsx, images)...",
    type=[
        "pdf",
        "pptx",
        "docx",
        "xlsx",
        "html",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "tiff",
    ],
    accept_multiple_files=True,
)

# Make sure we have a session‐scope holder for our conversions
if "conversions" not in st.session_state:
    st.session_state.conversions = {}

# Generate cache buster for current settings
current_cache_buster = generate_cache_buster(
    do_ocr,
    do_tables,
    match_cells,
    do_code,
    do_formulas,
    ocr_dpi,
    force_full_page_ocr,
    include_images,
    use_vlm_pipeline,
)

# Store the cache buster for each file to detect setting changes
if "cache_busters" not in st.session_state:
    st.session_state.cache_busters = {}

# Check if any files need reconversion due to setting changes
if uploaded and st.session_state.conversions:
    files_needing_reconversion = []
    for up in uploaded:
        key = up.name
        if (
            key in st.session_state.conversions
            and st.session_state.cache_busters.get(key) != current_cache_buster
        ):
            files_needing_reconversion.append(key)

    if files_needing_reconversion:
        st.info(
            f"⚙️ Settings changed - will reconvert: {', '.join(files_needing_reconversion)}"
        )

# As soon as the user picks files, loop & cache‐convert each
if uploaded:
    # Pre-read all files to avoid Streamlit file object issues
    files_data = []
    for up in uploaded:
        try:
            file_bytes = up.read()
            if not validate_file_size(up.name, file_bytes):
                continue
            files_data.append((up.name, file_bytes))
        except Exception as e:
            st.error(f"❌ Failed to read {up.name}: {e}")
            continue

    # Add progress bar for multiple files
    if len(files_data) > 1:
        progress_bar = st.progress(0)
        st.info(f"Processing {len(files_data)} files...")

    for i, (file_name, file_bytes) in enumerate(files_data):
        key = generate_file_key(file_name, file_bytes)

        # Check if file needs conversion (new file or settings changed)
        needs_conversion = (
            key not in st.session_state.conversions
            or st.session_state.cache_busters.get(key) != current_cache_buster
        )

        if needs_conversion:
            # Show spinner while converting
            with st.spinner(f"Please wait, converting {file_name}..."):
                try:
                    data = convert_document(
                        file_bytes,
                        file_name,
                        do_ocr,
                        do_tables,
                        match_cells,
                        do_code,
                        do_formulas,
                        ocr_dpi,
                        force_full_page_ocr,
                        include_images=include_images,
                        use_vlm_pipeline=use_vlm_pipeline,
                        _cache_buster=current_cache_buster,
                    )
                    st.session_state.conversions[key] = data
                    st.session_state.cache_busters[key] = current_cache_buster
                    st.success(f"✅ Finished converting {file_name}")
                except Exception as e:
                    st.error(f"❌ Conversion failed for {file_name}: {e}")
                    logger.error(f"Conversion error for {file_name}: {e}")
                    continue
        else:
            # File already converted with current settings
            st.info(f"Using cached conversion for {file_name}")

        # Update progress bar
        if len(files_data) > 1:
            progress_bar.progress((i + 1) / len(files_data))

    # Clean up progress bar
    if len(files_data) > 1:
        progress_bar.empty()

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

        # Show accelerator info
        if "accelerator" in data:
            st.caption(f"🚀 Accelerator: {data['accelerator']}")

        if data["ocr_settings"]:
            settings = data["ocr_settings"]
            st.caption(
                f"OCR Settings: DPI={settings['dpi']}, "
                f"Auto Language Detection, "
                f"Full Page OCR={'Yes' if settings['force_full_page_ocr'] else 'No'}"
            )
        if data["vlm_settings"]:
            vlm_settings = data["vlm_settings"]
            st.caption(
                f"🤖 VLM Pipeline: {vlm_settings['pipeline']} using {vlm_settings['model']}"
            )
        with st.expander("👁️ Preview Content", expanded=True):
            tab1, tab2, tab3, tab4 = st.tabs(["Markdown", "Text", "JSON", "HTML"])
            with tab1:
                preview_md, md_truncated = preview_text_content(
                    data["markdown"],
                    max_chars=preview_limit[2],
                    max_lines=preview_limit[1],
                )
                if md_truncated:
                    st.info(
                        f"📄 Preview shows first ~{preview_limit[1]} lines / {preview_limit[2]} characters. Download full file to see complete content."
                    )
                st.markdown(preview_md)
            with tab2:
                preview_txt, txt_truncated = preview_text_content(
                    data["text"], max_chars=preview_limit[2], max_lines=preview_limit[1]
                )
                if txt_truncated:
                    st.info(
                        f"📄 Preview shows first ~{preview_limit[1]} lines / {preview_limit[2]} characters. Download full file to see complete content."
                    )
                st.text(preview_txt)
            with tab3:
                preview_json, json_truncated = preview_json_content(
                    data["json"], max_chars=preview_limit[2]
                )
                if json_truncated:
                    st.info(
                        f"📄 Preview shows first ~{preview_limit[2]} characters. Download full file to see complete content."
                    )
                st.code(preview_json, language="json")
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
                        preview_html, html_truncated = preview_html_content(
                            html_content,
                            max_chars=int(
                                preview_limit[2] * 1.6
                            ),  # HTML needs more chars
                        )
                        if html_truncated:
                            st.info(
                                f"📄 Preview shows first ~{int(preview_limit[2] * 1.6)} characters. Download full file to see complete content."
                            )
                        st.components.v1.html(preview_html, height=400, scrolling=True)
                    else:
                        st.warning("No HTML file found in archive for preview.")
                except Exception as e:
                    st.error(f"Error extracting HTML preview: {e}")
else:
    for fname, data in st.session_state.conversions.items():
        st.markdown(f"---\n### {fname}")
        st.write(f"⏱ Converted in **{data['elapsed']:.2f}** seconds")

        # Show accelerator info
        if "accelerator" in data:
            st.caption(f"🚀 Accelerator: {data['accelerator']}")

        if data["ocr_settings"]:
            settings = data["ocr_settings"]
            st.caption(
                f"OCR Settings: DPI={settings['dpi']}, "
                f"Auto Language Detection, "
                f"Full Page OCR={'Yes' if settings['force_full_page_ocr'] else 'No'}"
            )
        if data["vlm_settings"]:
            vlm_settings = data["vlm_settings"]
            st.caption(
                f"🤖 VLM Pipeline: {vlm_settings['pipeline']} using {vlm_settings['model']}"
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
        with st.expander("👁️ Preview Content", expanded=True):
            tab1, tab2, tab3, tab4 = st.tabs(["Markdown", "Text", "JSON", "HTML"])
            with tab1:
                preview_md, md_truncated = preview_text_content(
                    data["markdown"],
                    max_chars=preview_limit[2],
                    max_lines=preview_limit[1],
                )
                if md_truncated:
                    st.info(
                        f"📄 Preview shows first ~{preview_limit[1]} lines / {preview_limit[2]} characters. Download full file to see complete content."
                    )
                st.markdown(preview_md)
            with tab2:
                preview_txt, txt_truncated = preview_text_content(
                    data["text"], max_chars=preview_limit[2], max_lines=preview_limit[1]
                )
                if txt_truncated:
                    st.info(
                        f"📄 Preview shows first ~{preview_limit[1]} lines / {preview_limit[2]} characters. Download full file to see complete content."
                    )
                st.text(preview_txt)
            with tab3:
                preview_json, json_truncated = preview_json_content(
                    data["json"], max_chars=preview_limit[2]
                )
                if json_truncated:
                    st.info(
                        f"📄 Preview shows first ~{preview_limit[2]} characters. Download full file to see complete content."
                    )
                st.code(preview_json, language="json")
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
                        preview_html, html_truncated = preview_html_content(
                            html_content,
                            max_chars=int(
                                preview_limit[2] * 1.6
                            ),  # HTML needs more chars
                        )
                        if html_truncated:
                            st.info(
                                f"📄 Preview shows first ~{int(preview_limit[2] * 1.6)} characters. Download full file to see complete content."
                            )
                        st.components.v1.html(preview_html, height=400, scrolling=True)
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
    col2.metric("Total Processing Time", f"{total_time:.2f}s")
    col3.metric("Average Time per File", f"{avg_time:.2f}s")

    # Add a clear cache button
    if st.button("Convert again!"):
        st.session_state.conversions = {}
        st.session_state.cache_busters = {}
        st.rerun()

# Add footer
current_year = time.strftime("%Y")
st.markdown("---")
st.caption(
    f"Docling Document Converter v1.6.0 | Copyright © 2025-{current_year} by bgeneto"
)
