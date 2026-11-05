"""
Alphabets of the dataset: Unicode code points and the associated font files.
"""

from glyphworld.core.paths import DATA_DIR


def extract_unicodes_as_ints(path, n):
    codes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # skip comments, empty lines, and header
            if not line or line.startswith("#") or line.lower().startswith("index"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            cp = parts[1]           # e.g. "U+4E00"
            value = int(cp[2:], 16) # -> 0x4E00
            codes.append(value)
            if len(codes) == n:
                break
    return codes


TGSCC_path = ( DATA_DIR / "raw" / ("TGSCC-Unicode.txt") )
n = 8105 #3500
cjk_unicodes = extract_unicodes_as_ints(TGSCC_path, n)


# Alphabets together with unicodes and font files.
alphabet_unicode_map = {
    "Ahom": (list(range(0x11700,0x1171B)),"NotoSerifAhom"),
    "Anatolian Hieroglyphs Full": (list(range(0x14400,0x14647)),"NotoSansAnatolianHieroglyphs"),
    "Arabic": (list(range(0x0621,0x063B))+list(range(0x0641,0x064b)),"NotoSansArabic"),
    "Armenian": (list(range(0x00531,0x00557)),"NotoSansArmenian"),
    "Avestan": (list(range(0x10B00,0x10B36)),"NotoSansAvestan"),
    "Balinese": (list(range(0x1B05,0x1B34)),"NotoSansBalinese"),
    "Batak": (list(range(0x1BC0,0x1BE6)),"NotoSansBatak"),
    "Bengali": (list(range(0x0985,0x098D))+[0x098F,0x0990]+list(range(0x0993,0x09A9))+list(range(0x09AA,0x09B1))+[0x09B2]+list(range(0x09B6,0x09BA)),"NotoSansBengali"),
    "Bhaiksuki": (list(range(0x11C00,0x11C09))+list(range(0x11C0A,0x11C2F)),"NotoSansBhaiksuki"),
    "Brahmi": (list(range(0x11005,0x11038)),"NotoSansBrahmi"),
    "Buginese": (list(range(0x1A00,0x1A17)),"NotoSansBuginese"),
    "Buhid": (list(range(0x1740,0x1752)),"NotoSansBuhid"),
    "Carian": (list(range(0x102A0,0x102D1)),"NotoSansCarian"),
    "Caucasian Albanian": (list(range(0x10530,0x10564)),"NotoSansCaucasianAlbanian"),
    "CJK Full": (cjk_unicodes,"NotoSansSC"),
    "Chakma": (list(range(0x11103,0x11127)),"NotoSansChakma"),
    "Cham": (list(range(0xAA00,0xAA29)),"NotoSansCham"),
    "Chorasmian": (list(range(0x10FB0,0x10FC5)),"NotoSansChorasmian"),
    "Coptic": ([0x2c80,0x2c82,0x2c84,0x2c86,0x2c88,0x2c8a,0x2c8c,0x2c8e,0x2c90,0x2c92,0x2c94,0x2c96,0x2c98,0x2c9a,0x2c9c,0x2c9e,0x2ca0,0x2ca2,0x2ca4,0x2ca6,0x2ca8,0x2caa,0x2cac,0x2cae,0x2cb0],"NotoSansCoptic"),
    "Cuneiform Full": (list(range(0x12000,0x1239A)),"NotoSansCuneiform"),
    "Cypriot Syllabary": (list(range(0x10800,0x10806))+list(range(0x1080A,0x10836))+[0x10808,0x10837,0x10838,0x1083C,0x1083F],"NotoSansCypriot"),
    "Cypro-Minoan": (list(range(0x012F90,0x012FF1)),"NotoSansCyproMinoan"),
    "Cyrillic": (list(range(0x00410,0x00430)),"NotoSans"),
    "Devanagari": (list(range(0x0904,0x093A)),"NotoSansDevanagari"),
    "Dogra": (list(range(0x11800,0x1182C)),"NotoSerifDogra"),
    "Egyptian Hieroglyphs Full": (list(range(0x13000, 0x1342F)),"NotoSansEgyptianHieroglyphs"),
    "Elder Futhark": ([0x016A0,0x016A2,0x016A6,0x016A8,0x016B1,0x016B2,0x016B7,0x016B9,0x016BA,0x016BE,0x016C1,0x016C3,0x016C7,0x016C8,0x016C9,0x016CA,0x016CF,0x016D2,0x016D6,0x016D7,0x016DA,0x016DC,0x016DE,0x016DF],"NotoSansRunic"),
    "Elymaic": (list(range(0x10FE0, 0x10FF6)),"NotoSansElymaic"),
    "Ge_ez": ([0x01200,0x01208,0x01210,0x01218,0x01220,0x01228,0x01230,0x01240,0x01260,0x01270,0x01280,0x01290,0x012A0,0x012A8,0x012C8,0x012D0,0x012D8,0x012E8,0x012F0,0x01308,0x01320,0x01330,0x01338,0x01340,0x01348,0x01350],"NotoSansEthiopic"),
    "Georgian Asomtavruli": (list(range(0x10A0,0x10C6)),"NotoSansGeorgian"),
    "Georgian Mkhedruli": (list(range(0x010D0,0x10F1)),"NotoSansGeorgian"),
    "Glagolitic": (list(range(0x02C00,0x02C2F)),"NotoSansGlagolitic"),
    "Gothic": (list(range(0x10330,0x1034B)),"NotoSansGothic"),
    "Grantha": (list(range(0x11305,0x1130D))+[0x1130F,0x11310]+list(range(0x11313,0x11329))+list(range(0x1132A,0x11331))+[0x11332,0x11333]+list(range(0x11335,0x1133A)),"NotoSansGrantha"),
    "Greek": (list(range(0x00391,0x003A2))+list(range(0x003A3,0x003AA)),"NotoSans"),
    "Gujarati": (list(range(0x0A85,0x0A8E))+list(range(0x0A8F,0x0A92))+list(range(0x0A93,0x0AA9))+list(range(0x0AAA,0x0AB1))+[0x0AB2,0x0AB3]+list(range(0x0AB5,0x0ABA)),"NotoSansGujarati"),
    "Gunjala Gondi": (list(range(0x11D60,0x11D66))+[0x11D67,0x11D68]+list(range(0x11D6A,0x11D8A)),"NotoSansGunjalaGondi"), # recent?
    "Gurmukhi": (list(range(0x0A05,0x0A0B))+[0x0A0F,0x0A10]+list(range(0x0A13,0x0A29))+list(range(0x0A2A,0x0A31))+[0x0A32,0x0A33,0x0A35,0x0A36,0x0A38,0x0A39],"NotoSansGurmukhi"),
    "Hangul Jamo": (list(range(0x1100,0x1113))+list(range(0x1161,0x1176))+list(range(0x11A8,0x11C3)),"NotoSansKR"),
    "Hanunoo": (list(range(0x1720,0x1732)),"NotoSansHanunoo"),
    "Hatran Aramaic": (list(range(0x108E0,0x108F3))+list(range(0x108F4,0x108F6)),"NotoSansHatran"),
    "Hebrew": (list(range(0x005D0,0x005EB)),"NotoSansHebrew"),
    "Hiragana": ([0x3042,0x3044,0x3046,0x3048]+list(range(0x304A,0x3063))+list(range(0x3064,0x3083))+[0x3084,0x3086]+list(range(0x3088,0x308E))+list(range(0x308F,0x3095)),"NotoSansJP"),
    "Imperial Aramaic": (list(range(0x10840,0x10856)),"NotoSansImperialAramaic"),
    "Javanese": (list(range(0xA984,0xA9B3)),"NotoSansJavanese"),
    "Kaithi": (list(range(0x11083,0x110B0)),"NotoSansKaithi"),
    "Kanbun": (list(range(0x3190,0x31A0)),"NotoSansJP"),
    "Kannada": (list(range(0x0C85,0x0C8D))+list(range(0x0C8E,0x0C91))+list(range(0X0C92,0x0CA9))+list(range(0x0CAA,0x0CB4))+list(range(0x0CB5,0x0CBA)),"NotoSansKannada"),
    "Katakana": ([0x30A2,0x30A4,0x30A6,0x30A8]+list(range(0x30AA,0x30C3))+list(range(0x30C4,0x30E3))+[0x30E4,0x30E6]+list(range(0x30E8,0x30EE))+list(range(0x30EF,0x30F5))+list(range(0x30F7,0x30FB)),"NotoSansJP"),
    "Kawi": (list(range(0x11F04,0x11F11))+list(range(0x11F12,0x11F34)),"NotoSansKawi"),
    "Kharoshthi": ([0x10A00]+list(range(0x10A10,0x10A14))+list(range(0x10A15,0x10A18))+list(range(0x10A19,0x10A36)),"NotoSansKharoshthi"),
    "Khmer": (list(range(0x1780,0x17A3))+list(range(0x17A5,0x17B4)),"NotoSansKhmer"),
    "Khojki": (list(range(0x11200,0x11212))+list(range(0x11213,0x1122C)),"NotoSansKhojki"),
    "Khudawadi": (list(range(0x112B0,0x112DF)),"NotoSansKhudawadi"),
    "Lao": ([0x0E81,0x0E82,0x0E84]+list(range(0x0E86,0x0E8B))+list(range(0x0E8C,0x0EA4))+[0x0EA5]+list(range(0x0EA7,0x0EAF)),"NotoSansLao"),
    "Latin": (list(range(0x00041,0x0005B)),"NotoSans"),
    "Lepcha": (list(range(0x1C00,0x1C24)),"NotoSansLepcha"),
    "Limbu": (list(range(0x1900,0x191F)),"NotoSansLimbu"),
    "Linear A Full": (list(range(0x10600,0x106A0)),"NotoSansLinearA"),
    "Linear B Full": (list(range(0x10000,0x1000C))+list(range(0x1000D,0x10027))+list(range(0x10028,0x1003B))+list(range(0x1003C,0x1003E))+list(range(0x1003F,0x10040))+list(range(0x10080,0x100FB)),"NotoSansLinearB"),
    "Lycian": (list(range(0x10280,0x1029D)),"NotoSansLycian"),
    "Lydian": (list(range(0x10920,0x1093A)),"NotoSansLydian"),
    "Mahajani": (list(range(0x11150,0x11173)),"NotoSansMahajani"),
    "Makasar": (list(range(0x11EE0,0x11EF3)),"NotoSerifMakasar"),
    "Malayalam": (list(range(0x0D05,0x0D0D))+list(range(0x0D0E,0x0D11))+list(range(0x0D12,0x0D3B)),"NotoSansMalayalam"),
    "Mandaic": (list(range(0x0840,0x0859)),"NotoSansMandaic"),
    "Manichaean": (list(range(0x10AC0,0x10AE5)),"NotoSansManichaean"),
    "Marchen": (list(range(0x11C72,0x11C90)),"NotoSansMarchen"),
    "Meetei Mayek": (list(range(0xABC0,0xABE3)),"NotoSansMeeteiMayek"),
    "Meroitic Cursive": (list(range(0x109A0,0x109B8)),"NotoSansMeroitic"),
    "Meroitic Hieroglyphs": (list(range(0x10980,0x109A0)),"NotoSansMeroitic"),
    "Modi": (list(range(0x11600,0x11630)),"NotoSansModi"),
    "Mongolian": (list(range(0x1820,0x1843)),"NotoSansMongolian"),
    "Multani": (list(range(0x11280,0x11287))+[0x11288]+list(range(0x1128A,0x1128E))+list(range(0x1128F,0x1129E))+list(range(0x1129F,0x112A9)),"NotoSansMultani"), # recent ?
    "Myanmar": (list(range(0x1000,0x102B)),"NotoSansMyanmar"), # pb
    "Nabataean": (list(range(0x10880,0x1089F)),"NotoSansNabataean"),
    "Nandinagari": (list(range(0x119A0,0x119A8))+list(range(0x119AA,0x119D1)),"NotoSansNandinagari"),
    "Newa": (list(range(0x11400,0x11435)),"NotoSansNewa"),
    "Ogham": (list(range(0x1681,0x169B)),"NotoSansOgham"),
    "Old Hungarian": (list(range(0x10C80,0x10CB3)),"NotoSansOldHungarian"),
    "Old Italic": (list(range(0x10300,0x1031B)),"NotoSansOldItalic"),
    "Old North Arabian": (list(range(0x10A80,0x10A9D)),"NotoSansOldNorthArabian"),
    "Old Permic": (list(range(0x10350,0x10376)),"NotoSansOldPermic"),
    "Old Persian": (list(range(0x103A0,0x103C4)),"NotoSansOldPersian"),
    "Old Sogdian": ([0x10F00,0x10F02,0x10F04,0x10F05,0x10F07,0x10F08,0x10F09,0x10F0A,0x10F0B,0x10F0C,0x10F0D,0x10F0E,0x10F11,0x10F14,0x10F15,0x10F18,0x10F19,0x10F1A],"NotoSansOldSogdian"),
    "Old South Arabian": (list(range(0x10A60,0x10A7D)),"NotoSansOldSouthArabian"),
    "Old Turkic Orkhon": ([0x10c00,0x10c03,0x10c06,0x10c07,0x10c09,0x10c0B,0x10c0D,0x10c0F,0x10c11,0x10c13,0x10c14,0x10c16,0x10c18,0x10c1A,0x10c1C,0x10c1E,0x10c20,0x10c21,0x10c22,0x10c23,0x10c24,0x10c26,0x10c28,0x10c2a,0x10c2d,0x10c2f,0x10c30,0x10c31,0x10c32,0x10c34,0x10c36,0x10c38,0x10c3a,0x10c3c,0x10c3d,0x10c3e,0x10c3f,0x10c41,0x10c43,0x10c45,0x10c47,0x10c48],"NotoSansOldTurkic"),
    "Old Turkic Yenisei": ([0x10c01,0x10c02,0x10c04,0x10c05,0x10c08,0x10c0a,0x10c0c,0x10c0e,0x10c10,0x10c12,0x10c15,0x10c17,0x10c19,0x10c1b,0x10c1d,0x10c1f,0x10c25,0x10c27,0x10c29,0x10c2b,0x10c2c,0x10c2e,0x10c33,0x10c35,0x10c37,0x10c39,0x10c3b,0x10c40,0x10c42,0x10c44,0x10c46],"NotoSansOldTurkic"),
    "Old Uyghur": (list(range(0x10F70,0x10F82)),"NotoSerifOldUyghur"),
    "Oriya": (list(range(0x0B05,0x0B0D))+[0x0B0F,0x0B10]+list(range(0x0B13,0x0B29))+list(range(0x0B2A,0x0B31))+[0x0B32,0x0B33]+list(range(0x0B35,0x0B3A)),"NotoSansOriya"),
    "Pahlavi Inscriptional": (list(range(0x10B60,0x10B73)),"NotoSansInscriptionalPahlavi"),
    "Pahlavi Psalter": (list(range(0x10b80,0x10b92)),"NotoSansPsalterPahlavi"),
    "Palmyrene": (list(range(0x10860,0x10877)),"NotoSansPalmyrene"),
    "Parthian Inscriptional": (list(range(0x10B40,0x10B56)),"NotoSansInscriptionalParthian"),
    "Phags-pa": (list(range(0xA840,0xA867)),"NotoSansPhagsPa"),
    "Phaistos Disc": (list(range(0x101D0,0x101FD)),"NotoSansSymbols2"),
    "Phoenician": (list(range(0x10900,0x10916)),"NotoSansPhoenician"),
    "Rejang": (list(range(0xA930,0xA947)),"NotoSansRejang"),
    "Samaritan": (list(range(0x0800,0x0816)),"NotoSansSamaritan"),
    "Sharada": (list(range(0x11183,0x111B3)),"NotoSansSharada"),
    "Siddham": (list(range(0x11580,0x115AF)),"NotoSansSiddham"),
    "Sinhala": (list(range(0x0D85,0x0D97))+list(range(0x0D9A,0x0DB2))+list(range(0x0DB3,0x0DBC))+[0x0DBD]+list(range(0x0DC0,0x0DC7)),"NotoSansSinhala"),
    "Sogdian": (list(range(0x10F30,0x10F46)),"NotoSansSogdian"),
    "Soyombo": (list(range(0x11A5C,0x11A84))+[0x11A50],"NotoSansSoyombo"),
    "Sundanese": (list(range(0x1B83,0x1BA1)),"NotoSansSundanese"),
    "Syloti Nagri": (list(range(0xA807,0xA80B))+list(range(0xA80C,0xA823)),"NotoSansSylotiNagri"),
    "Syriac": ([0x0710,0x0712,0x0713,0x0715]+list(range(0x0717,0x071C))+list(range(0x071D,0x0724))+list(range(0x0725,0x0727))+list(range(0x0728,0x0730)),"NotoSansSyriac"),
    "Tagalog": (list(range(0x1700,0x1712)),"NotoSansTagalog"),
    "Tagbanwa": (list(range(0X1760,0x176D))+list(range(0x176E,0x1771)),"NotoSansTagbanwa"),
    "Tai Le": (list(range(0x1950,0x196E)),"NotoSansTaiLe"),
    "Tai Tham": (list(range(0x1A20,0x1A55)),"NotoSansTaiTham"),
    "Tai Viet": (list(range(0xAA80,0xAAB0)),"NotoSansTaiViet"),
    "Takri": (list(range(0x11680,0x116AB)),"NotoSansTakri"),
    "Tamil": (list(range(0x0B85,0x0B8B))+list(range(0x0B8E,0x0B91))+list(range(0x0B92,0x0B96))+[0x0B99,0x0B9A,0x0B9C,0x0B9E,0x0B9F,0x0BA3,0x0BA4]+list(range(0x0BA8,0x0BAB))+list(range(0x0BAE,0x0BBA)),"NotoSansTamil"),
    "Tangut Full": (list(range(0x17000,0x18800)),"NotoSerifTangut"),
    "Telugu": (list(range(0x0C05,0x0C0D))+list(range(0x0C0E,0x0C11))+list(range(0x0C12,0x0C29))+list(range(0x0C2A,0x0C3A)),"NotoSansTelugu"),
    "Thaana": (list(range(0x0780,0x0798)),"NotoSansThaana"),
    "Thai": (list(range(0x0E01,0x0E30)),"NotoSansThai"),
    "Tibetan": (list(range(0x0F40,0x0F48))+list(range(0x0F49,0x0F6B)),"NotoSerifTibetan"),
    "Tifinagh": (list(range(0x2D30,0x2D68)),"NotoSansTifinagh"),
    "Tirhuta": (list(range(0x11481,0x114B0)),"NotoSansTirhuta"),
    "Todhri": (list(range(0x105C0,0x105F4)), "NotoSerifTodhri"),
    "Ugaritic": (list(range(0x10380,0x1039E)),"NotoSansUgaritic"),
    "Yezidi": (list(range(0x10E80,0x10EAA)),"NotoSerifYezidi"),
    "Zanabazar Square": (list(range(0x11A0B,0x11A33))+[0x11A00],"NotoSansZanabazarSquare"),
}

if __name__ == "__main__":
    nb_glyph_total = 0
    for alphabet in alphabet_unicode_map:
        nb_glyph = min(len(alphabet_unicode_map[alphabet][0]), 100)
        nb_glyph_total += nb_glyph
    print(f"nb_glyph_total: {nb_glyph_total}")
    
    
    def alphabet_map_to_latex_table(alphabet_unicode_map, caption=None, label=None):
        lines = []
        lines.append(r"\begin{table}[h]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{l r}")
        lines.append(r"\hline")
        lines.append(r"Script name & \# glyphs \\")
        lines.append(r"\hline")
    
        for script, (codepoints, font) in alphabet_unicode_map.items():
            nb_glyphs = len(codepoints)
            lines.append(f"{script} & {nb_glyphs} \\\\")
    
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
    
        if caption is not None:
            lines.append(rf"\caption{{{caption}}}")
        if label is not None:
            lines.append(rf"\label{{{label}}}")
    
        lines.append(r"\end{table}")
    
        return "\n".join(lines)
    
    
    # Example usage
    latex_table = alphabet_map_to_latex_table(
        alphabet_unicode_map,
        caption="Number of glyphs per script",
        label="tab:script_glyph_counts",
    )
    
    print(latex_table)
