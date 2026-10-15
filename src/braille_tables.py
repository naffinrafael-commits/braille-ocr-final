DOTS_TO_CHAR = {
    (1,): "a", (1, 2): "b", (1, 4): "c", (1, 4, 5): "d", (1, 5): "e",
    (1, 2, 4): "f", (1, 2, 4, 5): "g", (1, 2, 5): "h", (2, 4): "i", (2, 4, 5): "j",
    (1, 3): "k", (1, 2, 3): "l", (1, 3, 4): "m", (1, 3, 4, 5): "n", (1, 3, 5): "o",
    (1, 2, 3, 4): "p", (1, 2, 3, 4, 5): "q", (1, 2, 3, 5): "r", (2, 3, 4): "s", (2, 3, 4, 5): "t",
    (1, 3, 6): "u", (1, 2, 3, 6): "v", (2, 4, 5, 6): "w", (1, 3, 4, 6): "x", (1, 3, 4, 5, 6): "y",
    (1, 3, 5, 6): "z",
    (1, 2, 3, 5, 6): "á", (1, 2, 3, 4, 5, 6): "é", (3, 4): "í", (3, 4, 6): "ó", (2, 3, 4, 5, 6): "ú",
    (1, 6): "â", (1, 2, 6): "ê", (1, 4, 5, 6): "ô", (1, 2, 4, 6): "à", (3, 4, 5): "ã",
    (2, 4, 6): "õ", (1, 2, 3, 4, 6): "ç", (1, 2, 5, 6): "ü",
    (2,): ",", (2, 3): ";", (2, 5): ":", (3,): ".", (2, 6): "?", (2, 3, 5): "!", (3, 6): "-",
}

CAPITAL = (4, 6)
NUMBER = (3, 4, 5, 6)
DIGITS = {"a": "1", "b": "2", "c": "3", "d": "4", "e": "5", "f": "6", "g": "7", "h": "8", "i": "9", "j": "0"}
CHAR_TO_DOTS = {v: k for k, v in DOTS_TO_CHAR.items()}
DIGIT_TO_LETTER = {v: k for k, v in DIGITS.items()}


def dots_to_unicode(dots):
    return chr(0x2800 + sum(1 << (d - 1) for d in dots))


def is_known(dots):
    dots = tuple(sorted(dots))
    return dots in DOTS_TO_CHAR or dots in (CAPITAL, NUMBER)


def decode_cells(cells):
    out = []
    capital = False
    number = False
    for cell in cells:
        if cell is None:
            out.append(" ")
            number = False
            capital = False
            continue
        dots = tuple(sorted(cell))
        if dots == CAPITAL:
            capital = True
            continue
        if dots == NUMBER:
            number = True
            continue
        ch = DOTS_TO_CHAR.get(dots)
        if ch is None:
            out.append(dots_to_unicode(dots))
            continue
        if number and ch in DIGITS:
            out.append(DIGITS[ch])
            continue
        number = False
        if capital:
            ch = ch.upper()
            capital = False
        out.append(ch)
    return "".join(out)


def text_to_cells(text):
    cells = []
    number = False
    for ch in text:
        if ch == " " or ch == "\n":
            cells.append((ch, ()))
            number = False
            continue
        if ch.isdigit():
            if not number:
                cells.append(("#", NUMBER))
                number = True
            cells.append((ch, CHAR_TO_DOTS[DIGIT_TO_LETTER[ch]]))
            continue
        number = False
        low = ch.lower()
        if low not in CHAR_TO_DOTS:
            continue
        if ch != low:
            cells.append(("^", CAPITAL))
        cells.append((ch, CHAR_TO_DOTS[low]))
    return cells


def mask_to_dots(mask):
    return tuple(i + 1 for i in range(6) if mask & (1 << i))


def dots_to_mask(dots):
    return sum(1 << (d - 1) for d in dots)


CLASS_NAMES = ["".join(str(d) for d in mask_to_dots(m)) for m in range(1, 64)]


def dots_to_class(dots):
    m = dots_to_mask(dots)
    return m - 1 if 1 <= m <= 63 else None


def class_name_to_dots(name):
    name = str(name).strip()
    if name and all(c in "123456" for c in name):
        return tuple(sorted({int(c) for c in name}))
    low = name.lower()
    if len(low) == 1 and low in CHAR_TO_DOTS:
        return CHAR_TO_DOTS[low]
    return None




def dots_to_code(dots):
    return sum(1 << (d - 1) for d in dots)


def code_to_dots(code):
    return tuple(d for d in range(1, 7) if (code >> (d - 1)) & 1)


CLASS_CODES = list(range(1, 64))
CLASS_NAMES = ["".join(str(d) for d in code_to_dots(c)) for c in CLASS_CODES]


def class_index(dots):
    return dots_to_code(dots) - 1


def name_to_dots(name):
    name = str(name).strip()
    if name.isdigit() and all(ch in "123456" for ch in name):
        return tuple(sorted(set(int(ch) for ch in name)))
    low = name.lower()
    if low in CHAR_TO_DOTS:
        return CHAR_TO_DOTS[low]
    return None


def letter_to_dots(letter):
    return CHAR_TO_DOTS.get(str(letter).lower())
