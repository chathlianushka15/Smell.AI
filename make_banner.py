import math
import os

os.makedirs("assets", exist_ok=True)
GOLDEN = 2.399963            # the golden angle: how sunflowers arrange their seeds
W, H = 1200, 420

# A flower of dots on the right, one dot per fingerprint slot
n, cx, cy, radius = 700, 960, 210, 170
scale = radius / math.sqrt(n)
colours = ["#B97F7A", "#8A9A7B", "#CBBFA9"]
dots = []
for i in range(n):
    r = scale * math.sqrt(i + 0.5)
    a = i * GOLDEN
    x, y = cx + r * math.cos(a), cy + r * math.sin(a)
    size = 1.2 + 3.2 * i / n
    dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" '
                f'fill="{colours[i % 3]}" opacity="0.85"/>')

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<rect x="1" y="1" width="{W - 2}" height="{H - 2}" rx="28" fill="#F7F2E8" stroke="#E4D8C3" stroke-width="2"/>
<text x="76" y="112" font-family="Helvetica, Arial, sans-serif" font-size="16" letter-spacing="5" fill="#5F7152">AN AI THAT PREDICTS WHAT MOLECULES SMELL LIKE</text>
<text x="70" y="235" font-family="Georgia, 'Times New Roman', serif" font-size="128" font-weight="bold" fill="#4A382C">smell<tspan fill="#B97F7A">.</tspan>ai</text>
<text x="76" y="300" font-family="Georgia, 'Times New Roman', serif" font-size="34" font-style="italic" fill="#5F7152">Every molecule has a story.</text>
<text x="76" y="352" font-family="Helvetica, Arial, sans-serif" font-size="17" fill="#6E5A4B">Molecular machine learning  +  a simulated electronic nose</text>
{''.join(dots)}
</svg>"""

with open("assets/banner.svg", "w", encoding="utf-8") as f:
    f.write(svg)
print("Saved assets/banner.svg")