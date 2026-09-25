import sys, os
sys.path.insert(0, os.path.abspath('code/business_entity_resolution/src'))
from blocking import extract_blocking_keys
from normalization import normalize_name, normalize_address, extract_clean_numbers

s1_n = 'Vijay Trading Private Limited'
s1_a = 'J-3/299, Ground Floor Dda Flats Kalka Ji, New Delhi, South Delhi, Delhi'
c_n = 'विजय ट्रेडिंग प्राइवेट लिमिटेड'
c_a = 'J-D/3/299, GROUND FLOOR DDA FLATS KALKA JI, NEW DELHI, SOUTH DELHI, Delhi'

sys.stdout.reconfigure(encoding="utf-8")

pairs = [
    ("Vijay Trading Private Limited", "J-3/299, Ground Floor Dda Flats Kalka Ji, New Delhi, South Delhi, Delhi",
     "विजय ट्रेडिंग प्राइवेट लिमिटेड", "J-D/3/299, GROUND FLOOR DDA FLATS KALKA JI, NEW DELHI, SOUTH DELHI, Delhi"),
    ("Shakti Agro Limited", "C/O Gurnav Singh Saluja, Beside Zudio, Kultapara, Sadar, Sambalpur, Orissa",
     "ଶକ୍ତି ଆଗ୍ରୋ ଲିମିଟେଡ୍", "C/O GURNAV SINGH SALUJA, SADAR, Odisha"),
    ("Great Impex Private Limited", "6-2-101/5/C, Telangana, Hyderabad, Secunderabad, Lane Beside Centralview Apt New Bhoiguda",
     "గ్రేట్ ఇంపెక్స్ ప్రైవేట్ లిమిటెడ్", "6-2-101/5/C, Lane Beside Centralview Apt New Bhoiguda, Secunderabad, Hyderabad, TG"),
    ("Spicer Star Environmental Services LLC", "Fl 0, MO, Saint Louis, 9327 Atwood Drive",
     "Spicer-Star Environmental Seraices LLC", "932 Atwood Dr, Fl 0, Stlouis, Missouri"),
    ("Wojciechowski Federation", "610 Pennsylvania Avenue, Deer Lodge, MT",
     "Wojciechowski Ffdesatison", "Montana, Deer Lodge, 61 Pennsylvania Ave")
]

for s1_n, s1_a, c_n, c_a in pairs:
    k1 = extract_blocking_keys(s1_n, s1_a)
    k2 = extract_blocking_keys(c_n, c_a)
    print(f"\n--- {s1_n} vs {c_n} ---")
    print(f"Intersection ({len(k1 & k2)}): {k1 & k2}")
    if not (k1 & k2):
        print(f"S1 keys: {k1}")
        print(f"Cand keys: {k2}")

