import re
from fuzzywuzzy import process

def parse_nutrition(raw_text: str) -> dict:
    """
    Ekstrak nilai gizi dari raw_text OCR menggunakan Regex dan Fuzzy Matching.
    Target: sugar_g, carbs_g, calories, sodium_mg, fat_g
    """
    # Inisialisasi dictionary target dengan nilai default 0.0
    nutrition_data = {
        "sugar_g": 0.0,
        "carbs_g": 0.0,
        "calories": 0.0,
        "sodium_mg": 0.0,
        "fat_g": 0.0
    }
    
    raw_text = raw_text.lower()
    
    keywords = {
        "sugar_g": ["gula", "sugar", "gul", "sgr"],
        "carbs_g": ["karbohidrat", "carbs", "carbohydrate", "karbo"],
        "calories": ["kalori", "energi", "energy", "kkal", "cal"],
        "sodium_mg": ["natrium", "sodium", "garam", "na"],
        "fat_g": ["lemak", "fat", "lmk"]
    }
    
    # Tokenisasi berbasis spasi dan simbol pemisah
    tokens = re.split(r'[\s\-_:=]+', raw_text)
    
    for key, aliases in keywords.items():
        best_match_idx = -1
        best_score = 0
        
        # Cari token mana yang paling cocok
        for idx, token in enumerate(tokens):
            match_res = process.extractOne(token, aliases)
            if match_res:
                match_str, score = match_res
                # Fuzzy threshold 80
                if score > best_score and score >= 80:
                    best_score = score
                    best_match_idx = idx
        
        if best_match_idx != -1:
            found = False
            
            # 1. Cek apakah angka menempel pada kata (misal: "gula10g")
            num_match = re.search(r'(\d+[,\.]?\d*)', tokens[best_match_idx])
            if num_match:
                val_str = num_match.group(1).replace(',', '.')
                try:
                    nutrition_data[key] = float(val_str)
                    found = True
                except ValueError:
                    pass
            
            # 2. Jika tidak menempel, cek token di sebelahnya (sampai jarak 3 kata)
            if not found:
                for i in range(1, 4):
                    if best_match_idx + i < len(tokens):
                        candidate = tokens[best_match_idx + i]
                        num_match = re.search(r'(\d+[,\.]?\d*)', candidate)
                        if num_match:
                            val_str = num_match.group(1).replace(',', '.')
                            try:
                                nutrition_data[key] = float(val_str)
                                break
                            except ValueError:
                                continue
                                
    return nutrition_data
