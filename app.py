import html
import json
import math
import time
import urllib.parse
import urllib.request
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdDepictor
from pipeline import run_sniff, NAMES, smell_model, WORDS, gen, SAMPLES

st.set_page_config(page_title="smell.ai", page_icon="🌿", layout="centered")

# ------------------------------------------------------------------ data
FAMILIES = {
    "Floral": ["floral", "rose", "jasmin", "lily", "violet", "hyacinth", "geranium",
               "lavender", "hawthorn", "orris", "chamomile"],
    "Fruity": ["fruity", "apple", "pear", "peach", "apricot", "berry", "strawberry",
               "raspberry", "cherry", "grape", "banana", "pineapple", "melon", "plum",
               "tropical", "juicy", "black currant"],
    "Citrus": ["citrus", "lemon", "orange", "grapefruit", "bergamot"],
    "Green": ["green", "grassy", "leafy", "cucumber", "vegetable", "weedy", "tomato", "celery"],
    "Fresh and herbal": ["herbal", "mint", "cooling", "camphoreous", "pine", "terpenic",
                         "fresh", "clean", "ethereal", "aromatic"],
    "Woody and earthy": ["woody", "cedar", "sandalwood", "vetiver", "hay", "earthy",
                         "mushroom", "musty"],
    "Sweet and gourmand": ["sweet", "vanilla", "caramellic", "honey", "chocolate", "cocoa",
                           "coffee", "nutty", "almond", "hazelnut", "creamy", "milky",
                           "buttery", "coconut", "lactonic", "coumarin", "roasted", "malt"],
    "Spicy": ["spicy", "cinnamon", "clove", "anisic", "pungent", "tobacco", "balsamic", "warm"],
    "Smoky": ["smoky", "phenolic", "burnt", "leathery"],
    "Musk and powder": ["musk", "animalic", "powdery"],
    "Savoury": ["sulfurous", "alliaceous", "onion", "garlic", "meaty", "savory",
                "cabbage", "cheesy", "fatty", "fishy"],
}
WORD_INDEX = {str(w): i for i, w in enumerate(WORDS)}

# family -> (description, colour 1, colour 2, example molecules)
FAMILY_CARDS = {
    "Floral": ("Petals, blossom and soft powder.", "#F0CFCB", "#D4A5A0", ["linalool", "geraniol"]),
    "Fruity": ("Ripe orchard fruit and juicy sweetness.", "#F2C9A8", "#E3A08A",
               ["isoamyl acetate", "ethyl butyrate"]),
    "Citrus": ("Bright peel, zest and sparkle.", "#F1E3A0", "#D9C36A", ["limonene", "citral"]),
    "Green": ("Cut grass, leaves and garden air.", "#BFCDAE", "#8A9A7B", ["3-hexenol", "hexanal"]),
    "Fresh and herbal": ("Cool mint, pine and clean air.", "#CFE0D2", "#9DB8A6", ["menthol", "carvone"]),
    "Woody and earthy": ("Dry timber, resin and warm earth.", "#CDB59A", "#8E6F55",
                         ["beta-pinene", "camphor"]),
    "Sweet and gourmand": ("Vanilla, caramel and roasted warmth.", "#E8CFAF", "#B98B64",
                           ["vanillin", "coumarin"]),
    "Spicy": ("Clove, cinnamon and a warm bite.", "#E2A98A", "#B9674A", ["eugenol", "cinnamaldehyde"]),
    "Smoky": ("Wood fire, tar and burnt sweetness.", "#C9BFB5", "#7A6D63", ["guaiacol", "syringol"]),
}
EXTRA_FAMILIES = {
    "Musk and powder": ("Soft skin, powder and warmth.", "#E6D9DA", "#BFA9AE"),
    "Savoury": ("Roasted, brothy and sulfurous.", "#D9CDB0", "#A89A74"),
}

# =====================================================================
#  >>>  ADD YOUR PICTURES HERE  <<<
#  Paste the image address (Google Images -> open image -> right click
#  -> "Copy image address") between the quotes for each family.
#  Leave "" empty to keep the old colour for that family.
#  These pictures show on the "Scent families" cards AND on the
#  moodboard on the Discover page.
# =====================================================================
FAMILY_IMAGES = {
    "Floral": "https://www.gardenninja.co.uk/wp-content/uploads/2019/05/Cut-flowers.webp",
    "Fruity": "https://previews.123rf.com/images/aiselin/aiselin1901/aiselin190100048/115662462-healthy-fruit-platter-background-strawberries-raspberries-oranges-plums-apples-kiwis-grapes.jpg",
    "Citrus": "https://i.pinimg.com/736x/bb/d8/d5/bbd8d539d25e74d26a65cf7c80803a72.jpg",
    "Green": "https://png.pngtree.com/thumb_back/fh260/background/20240102/pngtree-vibrant-neon-tropical-plant-lush-green-leaves-fresh-foliage-and-abstract-image_13922172.png",
    "Fresh and herbal": "https://media.istockphoto.com/id/1635142635/photo/various-dried-medicinal-plants-herbs-and-flowers-on-an-old-wooden-background.jpg?s=612x612&w=0&k=20&c=BzFJPNPFpkmvDWoIxKzkH9xv0UGwi-l98mEJ5Swuce0=",
    "Woody and earthy": "https://thumbs.dreamstime.com/b/cozy-autumn-forest-trees-displaying-vibrant-shades-red-orange-yellow-landscape-359171430.jpg",
    "Sweet and gourmand": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSiFHAKQ4JFuwX3dYg5M2P4ULRgv77ijgCKBuEpxaA8aWNZWw21ECVSzvk&s=10",
    "Spicy": "https://media.istockphoto.com/id/477756915/photo/seamless-texture-with-spices-and-herbs.jpg?s=612x612&w=0&k=20&c=t9fckmk-oIJvwLgg-wA-DW6830JpAFtUmmVrr-JjmRc=",
    "Smoky": "https://static.vecteezy.com/system/resources/thumbnails/033/837/335/small/copy-space-background-dark-moody-scene-floor-black-wall-grey-patches-thick-smoke-billowing-around-ai-generative-photo.jpeg",
    "Musk and powder": "https://www.umangparticlescience.com/assets/images/products/outer-cards/Musk-Fragrance-Powder.webp",
    "Savoury": "https://thumbs.dreamstime.com/b/savoury-charcuterie-board-covered-meats-olives-peppers-berries-cheese-207167728.jpg",
}


def family_colours(f):
    entry = FAMILY_CARDS.get(f) or EXTRA_FAMILIES[f]
    return entry[1], entry[2]


def family_line(f):
    return (FAMILY_CARDS.get(f) or EXTRA_FAMILIES[f])[0]


def family_bg(f, angle=135):
    """CSS background for a family: its picture if you set one, with the colour
    gradient underneath as a fallback (shows if the picture fails to load)."""
    c1, c2 = family_colours(f)
    gradient = f"linear-gradient({angle}deg,{c1},{c2})"
    img = html.escape(FAMILY_IMAGES.get(f, "").strip())
    return f"url('{img}'), {gradient}" if img else gradient


# Every molecule from samples.json works offline, plus a few extras
LOCAL = {name.lower(): smi for group in SAMPLES.values() for name, smi in group.items()}
LOCAL.update({
    "ethanol": "CCO",
    "ethyl acetate": "CC(=O)OCC",
    "eugenol": "C=CCc1ccc(O)c(OC)c1",
    "geraniol": "CC(C)=CCCC(C)=CCO",
    "cinnamaldehyde": "O=CC=Cc1ccccc1",
    "benzyl acetate": "CC(=O)OCc1ccccc1",
    "camphor": "CC1(C)C2CCC1(C)C(=O)C2",
    "indole": "c1ccc2[nH]ccc2c1",
    "3-hexenol": "CCC=CCCO",
    "hexanal": "CCCCCC=O",
})
SMELL_NAMES = {n.lower(): n for n in SAMPLES}
GOLDEN = 2.399963   # the golden angle in radians: how sunflowers arrange their seeds
PAGES = ["Discover", "Scent families", "The science"]

# ------------------------------------------------------------------ style
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300..600;1,9..144,300..600&family=Jost:wght@300;400;500&display=swap');

:root {
  --ivory: #F7F2E8; --sand: #E4D8C3; --line: #CBBFA9;
  --sage: #8A9A7B; --sage-deep: #5F7152;
  --mocha: #4A382C; --soft: #6E5A4B;
  --rose: #D4A5A0; --rose-deep: #B97F7A;
  --serif: 'Fraunces', Georgia, serif;
  --sans: 'Jost', 'Segoe UI', sans-serif;
}
.stApp {
  background:
    radial-gradient(800px 520px at 92% 0%, rgba(138, 154, 123, 0.28), transparent 70%),
    radial-gradient(700px 520px at 0% 8%, rgba(212, 165, 160, 0.30), transparent 70%),
    #F7F2E8;
  color: var(--mocha);
}
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stStatusWidget"],
[data-testid="stDecoration"] { visibility: hidden; display: none; }
.block-container { max-width: 800px; padding-top: 1.2rem; padding-bottom: 4rem; }

/* two typefaces: Jost for everything, Fraunces for what speaks */
.stApp, .stApp div, .stApp p, .stApp span, .stApp label, .stApp button, .stApp input {
  font-family: var(--sans) !important;
}
.stApp .wordmark, .stApp .wordmark span, .stApp .headline, .stApp .big, .stApp .title,
.stApp .note .name, .stApp .fam-name, .stApp .stat .v, .stApp .finder-title,
.stApp .mood span { font-family: var(--serif) !important; }

/* wordmark */
.stApp .wordmark { text-align: center; font-size: clamp(3rem, 9.5vw, 5rem); font-weight: 500;
                   line-height: 1; letter-spacing: -0.025em; color: var(--mocha); padding: 0.3rem 0 1rem 0; }
.stApp .wordmark .dot { color: var(--rose-deep); }

/* navigation */
.st-key-page div[role="radiogroup"] { justify-content: center; gap: 2.8rem; flex-wrap: wrap;
                                      border-bottom: 1px solid var(--sand); }
.st-key-page label { padding: 0.6rem 0.1rem; margin-bottom: -1px; border-bottom: 2px solid transparent;
                     cursor: pointer; }
.st-key-page label > div:first-child { display: none; }
.st-key-page label p { font-size: 1.12rem; color: var(--soft); }
.st-key-page label:hover p { color: var(--mocha); }
.st-key-page label:has(input:checked) { border-bottom-color: var(--sage-deep); }
.st-key-page label:has(input:checked) p { color: var(--mocha); font-weight: 500; }

/* hero */
.hero-left { padding: 2.2rem 0 0.5rem 0; }
.eyebrow { font-size: 0.78rem; letter-spacing: 0.26em; font-weight: 500; color: var(--sage-deep); }
.stApp .headline { font-size: clamp(2.4rem, 6vw, 3.6rem); font-weight: 300; line-height: 1.08;
                   color: var(--mocha); margin: 0.9rem 0 1rem 0; }
.sub { font-weight: 300; font-size: 1.12rem; line-height: 1.7; color: var(--soft); max-width: 470px; }
.centered { text-align: center; margin-left: auto; margin-right: auto; }
.art { padding-top: 1rem; }
.art svg { width: 100%; height: auto; }

/* finder card */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255, 255, 255, 0.55); border: 1px solid var(--sand) !important;
  border-radius: 28px; padding: 0.6rem 0.8rem;
}
.stApp .finder-title { font-size: 1.9rem; font-weight: 400; color: var(--mocha); margin-top: 0.6rem; }
.fieldlabel { font-size: 0.85rem; font-weight: 500; color: var(--sage-deep); margin: 1rem 0 0.4rem 0; }
.stTextInput [data-baseweb="input"], .stTextInput [data-baseweb="base-input"] {
  background: rgba(255, 255, 255, 0.85) !important; border-radius: 999px !important;
}
.stTextInput [data-baseweb="input"] { border: 1px solid var(--line) !important; }
.stTextInput [data-baseweb="input"]:focus-within { border-color: var(--rose-deep) !important; }
.stTextInput input { background: transparent !important; color: var(--mocha) !important;
                     font-size: 1.05rem; padding: 0.95rem 1.5rem; }
.stTextInput input::placeholder { color: #9C8C7A; }

/* buttons: outlined by default, solid brown for the main action */
.stButton > button { background: transparent; color: var(--mocha); border: 1px solid var(--line);
                     border-radius: 999px; font-weight: 400; padding: 0.45rem 1rem; width: 100%; }
.stButton > button:hover { border-color: var(--rose-deep); color: var(--rose-deep); background: transparent; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: var(--mocha); color: var(--ivory); border: none; padding: 0.85rem 1.5rem;
  letter-spacing: 0.05em; font-weight: 500; font-size: 1.05rem;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
  background: var(--sage-deep); color: var(--ivory);
}
.stButton > button:focus-visible { outline: 2px solid var(--rose-deep); outline-offset: 3px; }

/* scent chips (nose): rose when selected. The Sniff button is a different shape and colour. */
.st-key-nose_pick div[role="radiogroup"] { justify-content: center; gap: 0.6rem; flex-wrap: wrap; }
.st-key-nose_pick label { border: 1px solid var(--line); border-radius: 999px; padding: 0.4rem 1.15rem;
                          background: rgba(255, 255, 255, 0.6); cursor: pointer; }
.st-key-nose_pick label:hover { border-color: var(--rose-deep); }
.st-key-nose_pick label > div:first-child { display: none; }
.st-key-nose_pick label p { color: var(--mocha); font-size: 0.98rem; }
.st-key-nose_pick label:has(input:checked) { background: var(--rose); border-color: var(--rose-deep); }
.st-key-nose_pick label:has(input:checked) p { font-weight: 500; }
.st-key-sniff button { border-radius: 18px; padding: 1rem 1.5rem; font-size: 1.15rem; }

/* text */
.rule { height: 1px; background: var(--sand); margin: 2.6rem 0 0 0; }
.stApp .title { font-style: italic; font-weight: 400; font-size: 1.5rem; color: var(--sage-deep);
                margin: 2.4rem 0 0.7rem 0; }
.stApp .big { font-size: 2.7rem; font-weight: 400; line-height: 1.15; color: var(--mocha); }
.body { font-size: 1rem; line-height: 1.75; color: var(--soft); max-width: 640px; }
.cap { font-size: 0.88rem; color: var(--soft); line-height: 1.6; margin-top: 0.5rem; }
.hint { text-align: center; color: var(--soft); margin-top: 1.2rem; font-size: 1rem; }
.badge { display: inline-block; font-size: 0.75rem; letter-spacing: 0.12em; font-weight: 500;
         border: 1px solid var(--sage-deep); color: var(--sage-deep); border-radius: 999px;
         padding: 0.15rem 0.8rem; margin-left: 0.8rem; vertical-align: middle; }
.chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.4rem; }
.chip { border: 1px solid var(--line); border-radius: 999px; padding: 0.3rem 1rem;
        font-size: 0.92rem; color: var(--soft); }
.chip.on { background: var(--rose); border-color: var(--rose-deep); color: var(--mocha); font-weight: 500; }

/* scent notes */
.note { display: flex; align-items: center; gap: 1rem; margin: 0.9rem 0; }
.stApp .note .name { font-size: 1.3rem; font-weight: 400; width: 10rem; }
.note .line { flex: 1; height: 16px; background: var(--sand); border-radius: 999px; overflow: hidden; }
.note .line div { height: 100%; background: linear-gradient(90deg, var(--rose), var(--rose-deep));
                  border-radius: 999px; }
.note .score { width: 3rem; text-align: right; font-size: 0.95rem; color: var(--soft); }

/* moodboard (pictures: the family name sits on a soft pill so it stays readable) */
.mood { display: flex; gap: 0.8rem; align-items: flex-end; margin-top: 1rem; }
.mood .blk { flex: 1; border-radius: 22px 22px 90px 90px; display: flex; align-items: flex-end;
             justify-content: center; padding-bottom: 1rem;
             background-size: cover; background-position: center; }
.stApp .mood span { font-size: 1.15rem; font-style: italic; color: var(--mocha);
                    background: rgba(247, 242, 232, 0.85); padding: 0.1rem 0.8rem; border-radius: 999px; }

.panel { text-align: center; }
.panel svg { width: 100%; height: auto; max-width: 380px; }
.trust { margin-top: 2.4rem; padding: 1.1rem 1.4rem; border-radius: 18px;
         background: rgba(138, 154, 123, 0.16); font-size: 0.95rem; line-height: 1.65; color: var(--soft); }

/* family cards and science stages */
.fam, .stage { border-radius: 22px; overflow: hidden; border: 1px solid var(--sand);
               background: rgba(255, 255, 255, 0.55); margin-bottom: 0.6rem; }
.fam .swatch { height: 110px; background-size: cover; background-position: center; }
.fam .inside, .stage .inside { padding: 0.9rem 1.1rem 1.1rem 1.1rem; }
.stApp .fam-name { font-size: 1.35rem; font-weight: 400; color: var(--mocha); }
.fam .desc { font-size: 0.92rem; color: var(--soft); line-height: 1.5; margin-top: 0.2rem; }
.fam .words { font-size: 0.8rem; color: var(--soft); margin-top: 0.7rem; line-height: 1.5; }
.stage { padding: 1.1rem 1.1rem 0.4rem 1.1rem; height: 100%; }
.stage .num { font-family: var(--serif) !important; font-size: 2rem; color: var(--rose-deep); }
.stage .icon svg { width: 100%; height: 70px; margin: 0.4rem 0; }
.stage h4 { font-family: var(--serif) !important; font-weight: 400; font-size: 1.2rem;
            color: var(--mocha); margin: 0.3rem 0; padding: 0; }
.stage p { font-size: 0.92rem; line-height: 1.6; color: var(--soft); }

/* nose results */
.stats { display: flex; gap: 0.8rem; margin-top: 0.6rem; flex-wrap: wrap; }
.stat { flex: 1; min-width: 150px; padding: 0.9rem 1.1rem; border-radius: 18px;
        border: 1px solid var(--sand); background: rgba(255, 255, 255, 0.55); }
.stat .k { font-size: 0.82rem; color: var(--soft); }
.stApp .stat .v { font-size: 1.7rem; color: var(--mocha); line-height: 1.2; }
.sensors { display: flex; gap: 12px; justify-content: center; margin-top: 0.6rem; }
.col { flex: 1; max-width: 64px; display: flex; flex-direction: column; align-items: center; }
.vialbox { height: 150px; width: 100%; display: flex; }
.vial { flex: 1; width: 100%; display: flex; align-items: flex-end; border: 1px solid var(--line);
        border-radius: 30px 30px 10px 10px; overflow: hidden; background: rgba(255,255,255,0.45); }
.liquid { width: 100%; background: linear-gradient(to top, var(--rose-deep), var(--rose)); }
.col .sn { font-size: 0.8rem; margin-top: 8px; color: var(--soft); }
.col .sv { font-size: 0.95rem; font-weight: 500; color: var(--mocha); }

/* footer */
.footer { text-align: center; margin-top: 4.5rem; padding-top: 2rem; border-top: 1px solid var(--sand); }
.footer .fine { font-size: 0.9rem; color: var(--soft); margin-top: 0.8rem; line-height: 1.7; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ drawings
def hero_art():
    leaves = ""
    for x, y, a, c in [(120, 392, -150, "#8A9A7B"), (110, 372, -118, "#5F7152"),
                       (290, 392, -30, "#8A9A7B"), (300, 368, -62, "#5F7152"),
                       (86, 330, -170, "#A9B79B"), (320, 332, -10, "#A9B79B")]:
        leaves += (f'<path d="M0,0 C26,-34 74,-34 98,0 C74,34 26,34 0,0Z" '
                   f'transform="translate({x},{y}) rotate({a})" fill="{c}" opacity="0.9"/>')
    blossoms = ""
    for x, y in [(330, 130), (352, 168), (312, 176)]:
        for k in range(5):
            ang = k * 72
            blossoms += (f'<ellipse cx="0" cy="-13" rx="8" ry="14" fill="#D4A5A0" '
                         f'transform="translate({x},{y}) rotate({ang})"/>')
        blossoms += f'<circle cx="{x}" cy="{y}" r="5" fill="#E4D8C3"/>'
    stem = '<path d="M312 176 C300 240 290 300 262 392" stroke="#5F7152" stroke-width="3" fill="none"/>'
    return (
        '<svg viewBox="0 0 420 470" xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="Illustration of a perfume flask with botanical leaves and blossoms">'
        '<defs><linearGradient id="glass" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#F3E3DD"/><stop offset="1" stop-color="#D4A5A0"/>'
        '</linearGradient></defs>'
        '<circle cx="210" cy="250" r="190" fill="#E4D8C3" opacity="0.45"/>'
        + stem + blossoms + leaves +
        '<rect x="168" y="150" width="64" height="56" rx="8" fill="#CBBFA9"/>'
        '<rect x="156" y="96" width="88" height="62" rx="14" fill="#4A382C"/>'
        '<rect x="116" y="196" width="168" height="204" rx="48" fill="url(#glass)" stroke="#B97F7A" '
        'stroke-width="1.5"/>'
        '<rect x="144" y="256" width="112" height="92" rx="12" fill="#F7F2E8" opacity="0.9"/>'
        '<polygon points="200,268 213.9,276 213.9,292 200,300 186.1,292 186.1,276" fill="none" '
        'stroke="#5F7152" stroke-width="2"/><circle cx="200" cy="284" r="5" fill="#B97F7A"/>'
        '<text x="200" y="326" text-anchor="middle" font-family="Fraunces, Georgia, serif" '
        'font-size="20" fill="#4A382C">smell.ai</text></svg>'
    )


def spiral(fp=None, n=2048, size=400, radius=188, big=3.2, small=1.1):
    scale = radius / math.sqrt(n)
    dots = []
    for i in range(n):
        r = scale * math.sqrt(i + 0.5)
        a = i * GOLDEN
        x, y = size / 2 + r * math.cos(a), size / 2 + r * math.sin(a)
        on = True if fp is None else bool(fp[i])
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{big if on else small}" '
                    f'fill="{"#B97F7A" if on else "#CBBFA9"}"/>')
    return f'<svg viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">' + "".join(dots) + "</svg>"


ICON_FINGERPRINT = spiral(n=70, size=120, radius=56, big=4.2, small=4.2)
ICON_FOREST = ('<svg viewBox="0 0 120 70" xmlns="http://www.w3.org/2000/svg">'
               + "".join(f'<rect x="{10 + i * 20}" y="{70 - h}" width="12" height="{h}" rx="6" '
                         f'fill="{c}"/>' for i, (h, c) in enumerate(
                             [(26, "#CBBFA9"), (54, "#B97F7A"), (36, "#CBBFA9"),
                              (62, "#B97F7A"), (30, "#CBBFA9"), (44, "#8A9A7B")])) + '</svg>')
ICON_FAMILIES = ('<svg viewBox="0 0 120 70" xmlns="http://www.w3.org/2000/svg">'
                 '<circle cx="34" cy="35" r="26" fill="#D4A5A0" opacity="0.85"/>'
                 '<circle cx="62" cy="35" r="26" fill="#8A9A7B" opacity="0.75"/>'
                 '<circle cx="88" cy="35" r="26" fill="#E3A08A" opacity="0.7"/></svg>')


def structure_svg(mol):
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)
    Chem.Kekulize(m, clearAromaticFlags=True)
    conf = m.GetConformer()
    pts = [(conf.GetAtomPosition(i).x, -conf.GetAtomPosition(i).y) for i in range(m.GetNumAtoms())]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pad = 0.9
    minx, miny = min(xs) - pad, min(ys) - pad
    w, h = max(xs) + pad - minx, max(ys) + pad - miny
    scale = min(380 / w, 280 / h, 60)
    W, H = w * scale, h * scale

    def pos(i):
        return (pts[i][0] - minx) * scale, (pts[i][1] - miny) * scale

    colours = {"O": "#B97F7A", "N": "#5F7152", "S": "#B8923F"}
    labelled = {a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() != "C"}
    parts = []
    for bond in m.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        (x1, y1), (x2, y2) = pos(a), pos(b)
        length = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        nx, ny = -uy, ux
        if a in labelled:
            x1, y1 = x1 + ux * 13, y1 + uy * 13
        if b in labelled:
            x2, y2 = x2 - ux * 13, y2 - uy * 13
        order = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
                 Chem.BondType.TRIPLE: 3}.get(bond.GetBondType(), 1)
        for k in range(order):
            off = (k - (order - 1) / 2) * 5
            parts.append(f'<line x1="{x1 + nx * off:.1f}" y1="{y1 + ny * off:.1f}" '
                         f'x2="{x2 + nx * off:.1f}" y2="{y2 + ny * off:.1f}" '
                         f'stroke="#4A382C" stroke-width="2" stroke-linecap="round"/>')
    for atom in m.GetAtoms():
        if atom.GetIdx() in labelled:
            x, y = pos(atom.GetIdx())
            hs = atom.GetTotalNumHs()
            label = atom.GetSymbol() + ("" if hs == 0 else "H" if hs == 1 else f"H{hs}")
            colour = colours.get(atom.GetSymbol(), "#4A382C")
            parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
                         f'dominant-baseline="central" font-family="Jost, sans-serif" '
                         f'font-size="17" font-weight="500" fill="{colour}">{label}</text>')
    return (f'<svg viewBox="0 0 {W:.0f} {H:.0f}" xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-label="Chemical structure of the molecule">' + "".join(parts) + "</svg>")


# ------------------------------------------------------------------ logic
def predict_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    fp = gen.GetFingerprintAsNumPy(mol)
    probs = smell_model.predict_proba(fp.reshape(1, -1))
    chance = np.array([p[0][1] for p in probs])
    return mol, fp, chance


def family_scores(chance):
    scores = {}
    for family, members in FAMILIES.items():
        values = [chance[WORD_INDEX[w]] for w in members if w in WORD_INDEX]
        scores[family] = max(values) if values else 0.0
    return scores


@st.cache_data(show_spinner="Looking up the molecule...")
def pubchem_smiles(name):
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           + urllib.parse.quote(name) + "/property/CanonicalSMILES/JSON")
    with urllib.request.urlopen(url, timeout=10) as response:
        data = json.load(response)
    props = data["PropertyTable"]["Properties"][0]
    for key, value in props.items():
        if "SMILES" in key:
            return value
    raise ValueError("no structure found")


def resolve(text):
    """Turn what the person typed into (display name, SMILES), or None."""
    key = text.lower()
    if key in LOCAL:
        return text.capitalize(), LOCAL[key]
    if " " not in text:
        mol = Chem.MolFromSmiles(text)
        if mol is not None and mol.GetNumAtoms() > 0:
            return "Your molecule", text
    try:
        return text.capitalize(), pubchem_smiles(key)
    except Exception:
        return None


def render_results(label, smiles):
    outcome = predict_molecule(smiles)
    if outcome is None:
        st.markdown('<p class="hint">I found that molecule but couldn’t read its structure. '
                    'Try another one.</p>', unsafe_allow_html=True)
        return
    mol, fp, chance = outcome
    top = np.argsort(chance)[::-1][:6]
    fam = family_scores(chance)
    ranked = sorted(fam, key=fam.get, reverse=True)
    best = ranked[0]

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="title">Scent profile for {html.escape(label)}</div>'
                f'<div class="big">{" · ".join(str(WORDS[i]).capitalize() for i in top[:3])}</div>'
                f'<p class="body" style="margin-top:0.8rem;">Mostly a <b>{best.lower()}</b> character: '
                f'{family_line(best).lower()}</p>', unsafe_allow_html=True)

    blocks = ""
    for height, family in zip([210, 165, 125], ranked[:3]):
        blocks += (f'<div class="blk" style="height:{height}px;background:{family_bg(family, 160)};'
                   f'background-size:cover;background-position:center">'
                   f'<span>{family}</span></div>')
    st.markdown(f'<div class="mood">{blocks}</div>'
                '<div class="cap">Moodboard: the pictures stand for the three strongest predicted scent '
                'families. It is illustrative, not a photograph of the molecule.</div>',
                unsafe_allow_html=True)

    chips = "".join(f'<span class="chip{" on" if f == best else ""}">{f}</span>' for f in ranked)
    st.markdown(f'<div class="title">Scent family</div><div class="chips">{chips}</div>',
                unsafe_allow_html=True)

    strongest = max(float(chance[top[0]]), 0.01)
    notes = ""
    for i in top:
        width = min(float(chance[i]) / strongest, 1.0) * 100
        notes += (f'<div class="note"><div class="name">{str(WORDS[i]).capitalize()}</div>'
                  f'<div class="line"><div style="width:{width:.0f}%"></div></div>'
                  f'<div class="score">{chance[i]:.2f}</div></div>')
    st.markdown('<div class="title">Scent notes</div>' + notes +
                '<div class="cap">The number is the share of the model’s 100 decision trees that '
                'vote for that note. It is a score for ranking, not a calibrated probability.</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="title">Molecular structure</div>'
                    f'<div class="panel">{structure_svg(mol)}</div>'
                    '<div class="cap">The molecule as chemists draw it.</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="title">Fingerprint</div>'
                    f'<div class="panel">{spiral(fp)}</div>'
                    f'<div class="cap">{int(fp.sum())} of 2,048 slots switched on. Each rose dot is a '
                    'chemical fragment found in the molecule.</div>', unsafe_allow_html=True)

    st.markdown('<div class="trust">An experimental prediction, not a laboratory measurement. The '
                'fingerprint describes structure, not smell: the model has only learned statistical '
                'links between structure and the labels perfume experts gave 4,983 molecules.</div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------ memory
for key, value in {"page": "Discover", "query": "", "go": False, "result": None,
                   "true_name": None, "nose_pick": NAMES[0]}.items():
    st.session_state.setdefault(key, value)
if "rng" not in st.session_state:
    st.session_state.rng = np.random.default_rng()


def pick(name):
    st.session_state.query = name
    st.session_state.go = True
    st.session_state.page = "Discover"


def explore():
    st.session_state.go = True


def clear():
    st.session_state.query = ""
    st.session_state.go = False


# ------------------------------------------------------------------ header and nav
st.markdown('<div class="wordmark">smell<span class="dot">.</span>ai</div>', unsafe_allow_html=True)
st.radio("Navigate", PAGES, key="page", horizontal=True, label_visibility="collapsed")
page = st.session_state.page

# ================================================================== DISCOVER
if page == "Discover":
    hl, hr = st.columns([1.15, 1])
    with hl:
        st.markdown(
            '<div class="hero-left"><div class="eyebrow">THE ART OF MOLECULAR SCENT</div>'
            '<div class="headline">Every molecule has a story.</div>'
            '<p class="sub">From molecular structures to predicted scent profiles. '
            'Discover the stories hidden within chemistry.</p></div>', unsafe_allow_html=True)
    with hr:
        st.markdown(f'<div class="art">{hero_art()}</div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="finder-title">What does your molecule smell like?</div>'
                    '<p class="body">Enter a molecule name or its SMILES notation to uncover its '
                    'predicted scent.</p><div class="fieldlabel">Molecule name or SMILES</div>',
                    unsafe_allow_html=True)
        st.text_input("Molecule", key="query", on_change=explore,
                      placeholder="e.g. limonene or CC1=CCC(CC1)C(C)=C",
                      label_visibility="collapsed")
        st.button("Discover its scent ↗", type="primary", on_click=explore, key="discover",
                  use_container_width=True)
        st.markdown('<div class="fieldlabel">Not sure where to start? Try one:</div>',
                    unsafe_allow_html=True)
        for col, name in zip(st.columns(4), ["Vanillin", "Limonene", "Menthol", "Eugenol"]):
            with col:
                st.button(name, key=f"ex_{name}", on_click=pick, args=(name,))

    text = st.session_state.query.strip()
    if st.session_state.go and text:
        if text.lower() in SMELL_NAMES:
            whole = SMELL_NAMES[text.lower()]
            parts = ", ".join(SAMPLES[whole].keys())
            st.markdown(f'<p class="hint">“{html.escape(text)}” is a whole smell, made of many molecules. '
                        f'Try one of them: {html.escape(parts)}. Or sniff it on The science page.</p>',
                        unsafe_allow_html=True)
        else:
            found = resolve(text)
            if found is None:
                st.markdown(f'<p class="hint">I couldn’t find a molecule called “{html.escape(text)}”. '
                            'Check the spelling, try the chemical name (like limonene), or paste a '
                            'SMILES string.</p>', unsafe_allow_html=True)
            else:
                render_results(*found)
                st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
                _, mid, _ = st.columns([1, 1.2, 1])
                with mid:
                    st.button("Try another molecule", key="clear", on_click=clear)

# ================================================================== SCENT FAMILIES
elif page == "Scent families":
    st.markdown('<div class="hero-left" style="text-align:center;">'
                '<div class="eyebrow">SCENT FAMILIES</div>'
                '<div class="headline">Nine ways a molecule can smell.</div>'
                '<p class="sub centered">Pick an example and the model will read it for you.</p></div>',
                unsafe_allow_html=True)
    st.markdown('<div style="height:1.4rem"></div>', unsafe_allow_html=True)
    cards = list(FAMILY_CARDS.items())
    for start in range(0, len(cards), 3):
        for col, (family, (desc, c1, c2, examples)) in zip(st.columns(3), cards[start:start + 3]):
            with col:
                shown = [w for w in FAMILIES[family] if w in WORD_INDEX][:5]
                st.markdown(f'<div class="fam"><div class="swatch" '
                            f'style="background:{family_bg(family, 135)};'
                            f'background-size:cover;background-position:center"></div>'
                            f'<div class="inside"><div class="fam-name">{family}</div>'
                            f'<div class="desc">{desc}</div>'
                            f'<div class="words">{", ".join(shown)}</div></div></div>',
                            unsafe_allow_html=True)
                for ex in examples:
                    st.button(ex.capitalize(), key=f"fam_{family}_{ex}", on_click=pick, args=(ex,))

# ================================================================== THE SCIENCE
else:
    st.markdown('<div class="hero-left" style="text-align:center;">'
                '<div class="eyebrow">THE SCIENCE</div>'
                '<div class="headline">How smell.ai thinks.</div>'
                '<p class="sub centered">What happens after you press Discover, and how far to trust '
                'it.</p></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)

    stages = [
        ("1", ICON_FINGERPRINT, "Molecule to fingerprint",
         "Your molecule is read as a graph of atoms. Each chemical fragment it contains switches on "
         "one of 2,048 slots, called a Morgan fingerprint."),
        ("2", ICON_FOREST, "Fingerprint to scent",
         "A random forest learned from 4,983 molecules labelled by perfume experts how fragments go "
         "with 138 scent words. It scores every word for your molecule."),
        ("3", ICON_FAMILIES, "Scent to family",
         "The strongest words become the scent notes. We grouped the 138 words into 11 families "
         "ourselves, so the families are our own grouping, not part of the dataset."),
    ]
    for col, (num, icon, name, text) in zip(st.columns(3), stages):
        with col:
            st.markdown(f'<div class="stage"><div class="num">{num}</div>'
                        f'<div class="icon">{icon}</div><h4>{name}</h4><p>{text}</p></div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="title">How far to trust it</div>'
                '<p class="body"><b>Data.</b> 4,983 molecules from the GoodScents and Leffingwell '
                'collections, curated by the OpenPOM project. The scent words are labels written by '
                'experts. They are not measurements.<br><br>'
                '<b>Score.</b> 0.85 is the mean ROC-AUC across the 138 scent words, measured on 997 '
                'molecules (a random 20%) that were hidden during training. 0.5 is guessing and 1.0 is '
                'perfect. The model is strong on words tied to a clear chemical feature, such as '
                'sulfurous (0.98), and weak on subjective ones such as musty (0.66).<br><br>'
                '<b>Confidence.</b> Scores in the app are the share of decision trees that vote for a '
                'word. They rank the options but are not calibrated probabilities, so 90% does not mean '
                'the model is right nine times in ten.</p>', unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>'
                '<div class="title">Meet the electronic nose '
                '<span class="badge">SIMULATION</span></div>'
                '<p class="body">A real electronic nose works in three steps: a row of gas sensors, '
                'a signal, and a classifier that names the smell. Here the sensors are code, not '
                'hardware, so nothing is physically smelling anything. Choose a scent, then run the '
                'simulation.</p>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="fieldlabel">1. Choose a scent</div>', unsafe_allow_html=True)
        st.radio("Scent", NAMES, key="nose_pick", horizontal=True, label_visibility="collapsed")
        st.markdown('<div class="fieldlabel">2. Run the simulation</div>', unsafe_allow_html=True)
        if st.button("≋  Sniff this scent", key="sniff", type="primary", use_container_width=True):
            with st.spinner("Sampling the simulated sensors..."):
                time.sleep(0.7)   # a short pause so the run feels like a measurement
                st.session_state.result = run_sniff(st.session_state.nose_pick, st.session_state.rng)
                st.session_state.true_name = st.session_state.nose_pick
        st.markdown('<p class="cap" style="text-align:center;">Choose a scent first, then run the '
                    'simulation.</p>', unsafe_allow_html=True)

    result = st.session_state.result
    if result is None:
        st.markdown('<p class="hint">No sniff yet. Choose a scent and press the button.</p>',
                    unsafe_allow_html=True)
    else:
        guess, conf, truth = result["guess"], result["confidence"], st.session_state.true_name
        right = guess == truth
        st.markdown(
            '<div class="title">Result <span class="badge">SIMULATION</span></div>'
            '<div class="stats">'
            f'<div class="stat"><div class="k">Scent in the jar</div><div class="v">{truth}</div></div>'
            f'<div class="stat"><div class="k">Nose’s guess</div><div class="v">{guess}</div></div>'
            f'<div class="stat"><div class="k">Tree vote share</div><div class="v">{conf:.0%}</div></div>'
            '</div>'
            f'<div class="cap">{"Correct." if right else "Wrong."} '
            f'{"" if conf >= 0.70 else "Low vote share: treat this guess with care. "}'
            'The vote share is not a calibrated probability.</div>', unsafe_allow_html=True)

        cols = ""
        for i, value in enumerate(result["reading"]):
            height = max(min(abs(value - 1.0) / 0.5, 1.0) * 100, 4)
            cols += (f'<div class="col"><div class="vialbox"><div class="vial"><div class="liquid" '
                     f'style="height:{height:.0f}%"></div></div></div>'
                     f'<div class="sv">{value:.2f}</div><div class="sn">S{i + 1}</div></div>')
        st.markdown('<div class="title">The eight sensors</div>'
                    f'<div class="sensors">{cols}</div>'
                    '<div class="cap">Each number is a reading divided by clean air, so 1.00 means no '
                    'reaction. A fuller vial means a stronger reaction. Every simulated sensor responds '
                    'differently to four properties of the smell’s molecules: size, oiliness, oxygen and '
                    'sulfur. Random strength, humidity and noise are added to every sniff.</div>',
                    unsafe_allow_html=True)

        rows = ""
        for name, p in sorted(result["all_probs"].items(), key=lambda kv: kv[1], reverse=True):
            rows += (f'<div class="note"><div class="name">{name}</div>'
                     f'<div class="line"><div style="width:{p * 100:.0f}%"></div></div>'
                     f'<div class="score">{p:.2f}</div></div>')
        st.markdown('<div class="title">How the nose ranked all six scents</div>' + rows +
                    '<div class="cap">The classifier compares the eight readings with the patterns it '
                    'learned from 1,000 simulated sniffs. Similar-smelling scents (vanilla candle, '
                    'banana, wood smoke) look alike to it, which is where it makes mistakes.</div>',
                    unsafe_allow_html=True)

        notes = ""
        for word, p in result["words"]:
            notes += (f'<div class="note"><div class="name">{word.capitalize()}</div>'
                      f'<div class="line"><div style="width:{p * 100:.0f}%"></div></div>'
                      f'<div class="score">{p:.2f}</div></div>')
        st.markdown(f'<div class="title">What the smell model says about {guess.lower()}</div>' + notes +
                    '<div class="cap">The smell model describes the real molecules of the nose’s '
                    'guess. If the guess is wrong, so is the description.</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ footer
st.markdown(
    '<div class="footer"><div class="eyebrow">CHEMISTRY, REIMAGINED</div>'
    '<div class="fine">smell.ai © 2026<br>Predictions come from a random forest trained on 4,983 '
    'expert-labelled molecules (OpenPOM dataset). The electronic nose is a simulation.</div></div>',
    unsafe_allow_html=True)