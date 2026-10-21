from pathlib import Path
from PIL import Image, ImageDraw
import pypdfium2 as pdfium

root = Path(__file__).parent / 'word_qa'
for stem in ['main', 'supplement']:
    doc = pdfium.PdfDocument(str(root / (stem + '.pdf')))
    images = []
    folder = root / stem
    folder.mkdir(exist_ok=True)
    for i, page in enumerate(doc):
        im = page.render(scale=1.7).to_pil().convert('RGB')
        im.save(folder / f'page-{i+1}.png')
        images.append(im)
    for start in range(0, len(images), 6):
        sheet = Image.new('RGB', (1320, 1920), '#dddddd')
        draw = ImageDraw.Draw(sheet)
        for j, im in enumerate(images[start:start+6]):
            thumb = im.copy()
            thumb.thumbnail((640, 600))
            x, y = (j % 2) * 660, (j // 2) * 640
            draw.text((x + 15, y + 8), f'{stem} / {start+j+1}', fill='black')
            sheet.paste(thumb, (x + (660-thumb.width)//2, y+30))
        sheet.save(root / f'{stem}_sheet_{start//6+1}.png')
    print(stem, len(images), 'pages rendered')
