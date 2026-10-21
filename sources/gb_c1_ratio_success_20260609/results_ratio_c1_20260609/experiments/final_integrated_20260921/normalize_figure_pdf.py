from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'figures/final_v20_refined_20260921/fig1_v20_refined.pdf'
target = ROOT / 'GB-LCSS_Elsevier_LaTeX_final_integrated_20260921/figures/fig1.pdf'
reader = PdfReader(source)
writer = PdfWriter()
writer.pdf_header = '%PDF-1.5'
for page in reader.pages:
    writer.add_page(page)
writer.write(target)
assert len(PdfReader(target).pages) == 1
print(target)
