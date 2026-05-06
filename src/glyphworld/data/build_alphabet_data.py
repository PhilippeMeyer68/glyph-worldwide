# -*- coding: utf-8 -*-
"""
This script permits to create alphabets from ttf font files and unicodes.
"""


import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize

import glyphworld.data.alphabet_unicode as alphabet_unicode
from glyphworld.core.paths import ROOT, ALPHABETS_DIR, DATA_DIR
from glyphworld.core.utils import save_glyph_dictionary
from glyphworld.visualization.plot_utils import plot_a_glyph, plot_an_alphabet


def estimate_stroke_width(img):
    stroke = (img == 0)          # bool: True on ink
    skel = skeletonize(stroke)   # bool
    area = stroke.sum()
    length = skel.sum()
    return 0 if length == 0 else area / length


def render_text_image(letter, font, xsize, ysize, x_offset=0, y_offset=0):
    """Render the character onto a white canvas using the given font and offset."""
    img = Image.new("RGB", (xsize, ysize), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), text=letter, font=font, fill="black")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img = (img > 128).astype(np.uint8) * 255
    return img


def get_nonwhite_bounds(img):
    """Return the bounding box of non-white areas in the image."""
    x_nonwhite = np.where(np.any(img != 255, axis=0))[0]
    y_nonwhite = np.where(np.any(img != 255, axis=1))[0]
    return x_nonwhite, y_nonwhite


def create_glyph(char_code, xsize, ysize, initial_fontsize, font_path, letter_width=8):
    letter = chr(char_code)
    fontsize = initial_fontsize
    font = ImageFont.truetype(font_path, fontsize)
    img = render_text_image(letter, font, xsize, ysize)
    width_img = estimate_stroke_width(img)
    x_nonwhite, y_nonwhite = get_nonwhite_bounds(img)
    
    if width_img < letter_width:
        while (x_nonwhite[-1] < xsize - 1 and y_nonwhite[-1] < ysize - 1) and width_img < letter_width:
            font = ImageFont.truetype(font_path, fontsize)
            img = render_text_image(letter, font, xsize, ysize)
            width_img = estimate_stroke_width(img)
            x_nonwhite, y_nonwhite = get_nonwhite_bounds(img)
            fontsize += 1
    else:
        while fontsize > 1 and width_img > letter_width:
            font = ImageFont.truetype(font_path, fontsize)
            img = render_text_image(letter, font, xsize, ysize)
            width_img = estimate_stroke_width(img)
            x_nonwhite, y_nonwhite = get_nonwhite_bounds(img)
            fontsize -= 1

    while fontsize > 1 and (x_nonwhite[-1] < xsize - 1 and y_nonwhite[-1] < ysize - 1) == False:
        fontsize -= 1
        font = ImageFont.truetype(font_path, fontsize)
        img = render_text_image(letter, font, xsize, ysize)
        x_nonwhite, y_nonwhite = get_nonwhite_bounds(img)
    
    width = x_nonwhite[-1] - x_nonwhite[0] + 1
    height = y_nonwhite[-1] - y_nonwhite[0] + 1
    #print(letter, width, height)
    x_offset = (xsize - width) // 2 - x_nonwhite[0]
    y_offset = (ysize - height) // 2 - y_nonwhite[0]
    font = ImageFont.truetype(font_path, fontsize)
    final_img = render_text_image(letter, font, xsize, ysize, x_offset, y_offset)
    return final_img


def create_alphabet(
    package_root, alphabet_unicode_map, lang, xsize, ysize, fontsize, image_unsupported
):
    """
    Create the alphabet of a specified language and exports it as a numpy array.

    Parameters
    ----------
    alphabet_unicode_map : dict
        A dictionary containing alphabets for different languages.
    lang : str
        The language of the alphabet to create.
    xsize : int
        The width of the letters in the alphabet.
    ysize : int
        The height of the letters in the alphabet.
    fontsize : int
        The font size of the letters in the alphabet.
    image_unsupported : numpy.ndarray
        A reference image used to check if a generated glyph is unsupported.

    Returns
    -------
    numpy.ndarray
        An array containing the alphabet letters represented as numpy arrays.
    """
    
    range_alph = alphabet_unicode_map[lang]
    font = DATA_DIR / "raw" / "fonts" / f"{range_alph[1]}.ttf"

    X_glyphs = []
    tmp = 0
    for i in range_alph[0]:
        tmp += 1
        if tmp % 100 == 0:
            print(tmp)
        char = i
        image = create_glyph(char, xsize, ysize, fontsize, font)
        #plot_a_glyph(image)

        difference = cv2.subtract(image, image_unsupported)
        if not np.any(difference) == False:
            X_glyphs.append(image)
        else:
            # "tofu"/unsupported glyph detected: warn with unicode + char
            try:
                ch = chr(char)
            except Exception:
                ch = "<?>"
            print(
                f"[WARN] Unsupported/tofu glyph for alphabet='{lang}': "
                f"unicode=U+{char:04X} char='{ch}' (font='{font.name}')"
            )
    return np.array(X_glyphs)


if __name__ == "__main__":
    # We import the dictionary of alphabets and unicodes ranges.
    alphabet_unicode_map = alphabet_unicode.alphabet_unicode_map

    # Parameters of alphabets that will be exported.
    xsize, ysize = 105, 105  # Size of the images.
    fontsize = 51  # Size of the glyphs in the images.

    # Creation of the tofu image "unsupported" to test if the glyphs are well loaded
    char = 1
    font_arial = DATA_DIR / "raw" / "fonts" / "ArialUnicode.ttf"
    image_unsupported = create_glyph(char, xsize, ysize, fontsize, font_arial)

    # We give some statistics and we test an alphabet.
    print(
        "There are",
        len(alphabet_unicode_map),
        "alphabets in our database and",
        sum([len(alphabet_unicode_map[lang][0]) for lang in alphabet_unicode_map.keys()]),
        "glyphs."
    )

    alphabet = "Latin"
    print("We test the creation of the alphabet", alphabet, len(alphabet_unicode_map[alphabet][0]))
    alphabet_glyphs = create_alphabet(
        ROOT,
        alphabet_unicode_map,
        alphabet,
        xsize,
        ysize,
        fontsize,
        image_unsupported,
    )
    plot_an_alphabet(alphabet_glyphs)
    
    if 1 == 1:
        # We export all the alphabets as a single pickle dictionary.
        alphabets_dic = {}
        
        for alphabet in alphabet_unicode_map:
            print(f"{alphabet}: {len(alphabet_unicode_map[alphabet][0])}")
            alphabet_glyphs = create_alphabet(
                ROOT,
                alphabet_unicode_map,
                alphabet,
                xsize,
                ysize,
                fontsize,
                image_unsupported,
            )
            glyph_dict = {}
            unicode_codes = alphabet_unicode_map[alphabet][0]
    
            for idx, (img, code) in enumerate(zip(alphabet_glyphs, unicode_codes)):
                key = f"glyph_{idx:05d}_unicode_{code:05X}"
                glyph_dict[key] = img
    
            alphabets_dic[alphabet] = glyph_dict

        # save
        save_path = ALPHABETS_DIR / "alphabets_unicode.pkl"
        save_glyph_dictionary(alphabets_dic, save_path)
        print(f"[SAVED] {save_path}")
