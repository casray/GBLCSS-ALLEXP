from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'package' / 'GB-LCSS_PR_submission_credit_strict_20260930'
OUT = BASE / 'renders'
OUT.mkdir(exist_ok=True)
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
for name in ['main', 'supplementary']:
    doc = pdfium.PdfDocument(ROOT / (name + '.pdf'))
    thumbnails = []
    for i in range(len(doc)):
        page = doc[i]
        bitmap = page.render(scale=1.5)
        img = bitmap.to_pil().convert('RGB')
        img.save(OUT / f'{name}-{i+1:02}.png')
        img.thumbnail((330, 430))
        tile = Image.new('RGB', (350, 465), '#dddddd')
        tile.paste(img, ((350-img.width)//2, 27))
        ImageDraw.Draw(tile).text((12, 4), f'{name} page {i+1}', fill='black', font=font)
        thumbnails.append(tile)
        page.close()
    for start in range(0, len(thumbnails), 9):
        sheet = Image.new('RGB', (1050, 1395), 'white')
        for j, thumb in enumerate(thumbnails[start:start+9]):
            sheet.paste(thumb, ((j % 3) * 350, (j // 3) * 465))
        sheet.save(OUT / f'{name}-sheet-{start//9+1}.png')
    doc.close()
print(OUT)
