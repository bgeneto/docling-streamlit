# Docling Web UI

A modern, user-friendly Web UI (Streamlit app) for converting documents (PDF, DOCX, PPTX, images, etc.) using the [Docling](https://github.com/bgeneto/docling) library. Supports OCR, table detection, and exporting to Markdown, plain text, JSON, and HTML. Easily configure conversion options and download results in various formats.

---

## Features

- **Multi-format Support:** PDF, DOCX, PPTX, Markdown, and common image formats (PNG, JPG, GIF, BMP, TIFF)
- **OCR (Optical Character Recognition):** Extract text from scanned documents and images using Tesseract
- **Table Detection:** Extract and structure tables from documents
- **Image Extraction:** Include or exclude figures/images in output
- **Multiple Export Formats:** Markdown (with images), plain text, JSON, and self-contained HTML
- **Batch Processing:** Upload and convert multiple files at once
- **Customizable Pipeline:** Fine-tune OCR, table, and image settings
- **Download Options:** Download each format separately or all at once as a ZIP
- **Preview:** Instantly preview converted content in the browser

---

## Getting Started

### Prerequisites
- Python 3.8+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for OCR features)
- Docker (optional, for containerized deployment)
- NVIDIA GPU (optional, for faster conversion with GPU support)

### Installation

#### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/docling-streamlit.git
cd docling-streamlit
```

#### 2. Run via docker (recommended)
```bash
docker compose up -d
```

> NOTE: The Web UI is exposed on port 8501.

---

## Usage

1. **Upload Files:** Drag and drop or select files (PDF, DOCX, PPTX, images, etc.)
2. **Configure Options:**
   - Table detection, OCR, image extraction, and output settings
3. **Convert:** Wait for processing (progress shown)
4. **Preview & Download:** View results in Markdown, Text, JSON, or HTML. Download individually or as a ZIP.

---

## Screenshots

![Screenshot](https://github.com/user-attachments/assets/af89dc8e-fafe-4ef4-8ef3-e6237f1003ac)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Credits
- [Docling](https://github.com/bgeneto/docling)
- [Streamlit](https://streamlit.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

---

## Contributing

Pull requests and issues are welcome! Please open an issue to discuss your ideas or report bugs.

---

## Author

**bgeneto**
[GitHub](https://github.com/bgeneto)
