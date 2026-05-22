import re
from fuzzywuzzy import process

# Token satuan/unit dan kata umum yang tidak boleh di-match sebagai keyword gizi
UNIT_TOKENS = {
    "g", "gr", "mg", "kg", "ml", "l", "kkal", "kal", "cal",
    "%", "per", "total", "dari", "sajian", "saji",
    "jumlah", "nilai", "gizi", "informasi", "kemasan", "akg",
}


def parse_nutrition(raw_text: str) -> dict:
    """
    Ekstrak nilai gizi dari raw_text OCR menggunakan Regex dan Fuzzy Matching.
    Target: sugar_g, carbs_g, calories, sodium_mg, fat_g
    """
    nutrition_data = {
        "sugar_g": 0.0,
        "carbs_g": 0.0,
        "calories": 0.0,
        "sodium_mg": 0.0,
        "fat_g": 0.0
    }

    raw_text = raw_text.lower()

    keywords = {
        "sugar_g":   ["gula", "sugar", "gul", "sgr"],
        "carbs_g":   ["karbohidrat", "carbs", "carbohydrate", "karbo"],
        "calories":  ["kalori", "energi", "energy"],
        "sodium_mg": ["natrium", "sodium", "garam"],
        "fat_g":     ["lemak", "fat"],
    }

    tokens = re.split(r'[\s\-_:=()]+', raw_text)
    tokens = [t for t in tokens if t]

    # Pre-build: alias semua nutrisi LAIN untuk setiap key (untuk stop-lookahead)
    other_aliases = {}
    for key in keywords:
        flat = []
        for other_key, aliases in keywords.items():
            if other_key != key:
                flat.extend(aliases)
        other_aliases[key] = flat

    for key, aliases in keywords.items():
        best_match_idx = -1
        best_score = 0

        for idx, token in enumerate(tokens):
            # Skip satuan, kata umum, dan token terlalu pendek
            if token in UNIT_TOKENS or len(token) < 3:
                continue

            match_res = process.extractOne(token, aliases)
            if match_res:
                match_str, score = match_res
                if score > best_score and score >= 80:
                    best_score = score
                    best_match_idx = idx

        if best_match_idx == -1:
            continue

        found = False

        # 1. Angka menempel pada token keyword itu sendiri (misal: "gula10g")
        num_match = re.search(r'(\d+[,\.]?\d*)', tokens[best_match_idx])
        if num_match:
            val_str = num_match.group(1).replace(',', '.')
            try:
                nutrition_data[key] = float(val_str)
                found = True
            except ValueError:
                pass

        # 2. Cari angka di token sebelah kanan (jarak max 5 token)
        #    Berhenti jika ketemu keyword dari nutrisi BERBEDA (jangan ambil nilai baris lain)
        if not found:
            for i in range(1, 6):
                if best_match_idx + i >= len(tokens):
                    break

                candidate = tokens[best_match_idx + i]

                # Cek apakah token ini adalah keyword nutrisi dari kategori lain
                if len(candidate) >= 3 and candidate not in UNIT_TOKENS:
                    cross_match = process.extractOne(candidate, other_aliases[key])
                    if cross_match and cross_match[1] >= 80:
                        break  # masuk baris nutrisi lain, stop

                num_match = re.search(r'(\d+[,\.]?\d*)', candidate)
                if num_match:
                    val_str = num_match.group(1).replace(',', '.')
                    try:
                        nutrition_data[key] = float(val_str)
                        found = True
                        break
                    except ValueError:
                        continue

    return nutrition_data
