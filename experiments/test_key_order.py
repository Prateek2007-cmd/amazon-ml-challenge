import sys, os
sys.path.insert(0, os.path.abspath('code/business_entity_resolution/src'))
from blocking import extract_blocking_keys

name = "Spicer Star Environmental Services LLC"
addr = "Fl 0, MO, Saint Louis, 9327 Atwood Drive"

keys = extract_blocking_keys(name, addr)
print("Keys list order:")
for i, k in enumerate(keys):
    print(f"  {i}: {k[0]} -> {k}")
