# -*- coding: utf-8 -*-
"""
Plot utils.
"""

from matplotlib import pyplot as plt


def plot_a_glyph(glyph):
    plt.imshow(glyph, cmap="gray")
    plt.show()


def plot_an_alphabet(glyphs):
    for i in range(glyphs.shape[0]):
        glyph = glyphs[i]
        plt.imshow(glyph, cmap="gray")
        plt.show()


def plot_two_glyphs(glyph_1, glyph_2):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(glyph_1, cmap="gray")
    axes[0].axis("off")
    axes[1].imshow(glyph_2, cmap="gray")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
