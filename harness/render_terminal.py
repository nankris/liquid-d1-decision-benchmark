"""Render a UTF-8 text file (or '-' for stdin) as a terminal PNG using Pillow."""
import argparse
from pathlib import Path
import sys
import textwrap
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help="Input text file, or '-' for stdin")
    parser.add_argument('output')
    parser.add_argument('--title', default='d1 experiment harness')
    args = parser.parse_args()
    text = sys.stdin.read() if args.input == '-' else Path(args.input).read_text()
    lines = [part for line in text.expandtabs(4).splitlines() for part in (textwrap.wrap(line, width=110, replace_whitespace=False, drop_whitespace=False) or [''])] or ['']
    font = ImageFont.truetype('DejaVuSansMono.ttf', 15)
    title_font = ImageFont.truetype('DejaVuSansMono.ttf', 13)
    width = max(600, int(max(font.getlength(line) for line in lines)) + 48)
    height = 66 + len(lines) * 21
    image = Image.new('RGB', (width, height), '#1e1e2e')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 40), fill='#313244')
    for x, fill in zip([20, 40, 60], ['#ff5f57', '#febc2e', '#28c840']):
        draw.ellipse((x - 5, 15, x + 5, 25), fill=fill)
    title = textwrap.shorten(args.title, width=max(10, int((width - 180) / 8)), placeholder='…')
    draw.text((width / 2, 20), title, font=title_font, fill='#cdd6f4', anchor='mm')
    for i, line in enumerate(lines):
        draw.text((24, 53 + i * 21), line, font=font, fill='#cdd6f4')
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f'Wrote {args.output} ({width}×{height})')

if __name__ == '__main__':
    main()
