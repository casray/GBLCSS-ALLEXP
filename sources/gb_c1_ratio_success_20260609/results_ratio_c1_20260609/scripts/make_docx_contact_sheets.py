import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def make_sheets(folder: Path) -> None:
    pages = sorted(folder.glob("page-*.png"))
    for start in range(0, len(pages), 6):
        batch = pages[start : start + 6]
        thumbnails = []
        for page_number, path in enumerate(batch, start + 1):
            image = Image.open(path).convert("RGB")
            image.thumbnail((360, 510))
            canvas = Image.new("RGB", (380, 545), "white")
            canvas.paste(image, ((380 - image.width) // 2, 22))
            ImageDraw.Draw(canvas).text((10, 5), f"Page {page_number}", fill="black")
            thumbnails.append(canvas)

        sheet = Image.new("RGB", (380 * 3, 545 * 2), (230, 230, 230))
        for index, image in enumerate(thumbnails):
            sheet.paste(image, ((index % 3) * 380, (index // 3) * 545))
        output = folder / f"contact_{start + 1:02d}_{start + len(batch):02d}.png"
        sheet.save(output)
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="+")
    args = parser.parse_args()
    for folder in args.folders:
        make_sheets(Path(folder))
