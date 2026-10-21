from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE_PAGES = ROOT / "_source_pages"


def crop_source_page(source_name: str, output_name: str, box: tuple[int, int, int, int]) -> None:
    with Image.open(SOURCE_PAGES / source_name) as image:
        image.crop(box).save(ROOT / output_name, optimize=True)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def build_contact_sheet() -> None:
    entries = [
        ("1. GLNN - Fig. 2", "01_GLNN_Fig2_training_deployment.png"),
        ("2. GRACE - Fig. 1", "02_GRACE_Fig1_dual_graph_views.png"),
        ("3. SAGBC - Fig. 1", "03_SAGBC_Fig1_overall_structure.png"),
        ("4. SGBGC - Fig. 2", "04_SGBGC_Fig2_four_stage_pipeline.png"),
        ("5. G-GBC - Fig. 3", "05_GGBC_Fig3_multistep_pipeline.png"),
        ("6. MgCNL - Fig. 1", "06_MgCNL_Fig1_noisy_label_pipeline.png"),
        ("7. VQGraph - Fig. 2", "07_VQGraph_Fig2_tokenizer_distillation.png"),
        ("8. NOSMOG - Fig. 1", "08_NOSMOG_Fig1_teacher_student.png"),
        ("9. FF-G2M - Fig. 3", "09_FFG2M_Fig3_frequency_distillation.png"),
        ("10. NRGNN - Fig. 1", "10_NRGNN_Fig1_noisy_graph_framework.png"),
    ]

    column_width = 1160
    panel_height = 660
    margin = 32
    label_height = 52
    sheet = Image.new("RGB", (column_width * 2 + margin * 3, panel_height * 5 + margin * 6), "white")
    draw = ImageDraw.Draw(sheet)
    font = load_font(30)

    for index, (label, filename) in enumerate(entries):
        row, column = divmod(index, 2)
        x = margin + column * (column_width + margin)
        y = margin + row * (panel_height + margin)
        draw.text((x, y), label, fill="#20252b", font=font)

        with Image.open(ROOT / filename) as image:
            source = image.convert("RGB")
            target_width = column_width
            target_height = panel_height - label_height
            scale = min(target_width / source.width, target_height / source.height)
            resized = source.resize(
                (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
                Image.Resampling.LANCZOS,
            )
            paste_x = x + (target_width - resized.width) // 2
            paste_y = y + label_height + (target_height - resized.height) // 2
            sheet.paste(resized, (paste_x, paste_y))

    sheet.save(ROOT / "00_contact_sheet_top10.png", optimize=True)


if __name__ == "__main__":
    # Coordinates refer to the 360-dpi renderings of the original PDF pages.
    crop_source_page(
        "_NOSMOG_page4_360.png",
        "08_NOSMOG_Fig1_teacher_student.png",
        (455, 285, 2510, 1200),
    )
    crop_source_page(
        "_NRGNN_page4_360.png",
        "10_NRGNN_Fig1_noisy_graph_framework.png",
        (1475, 360, 3040, 1028),
    )
    build_contact_sheet()
