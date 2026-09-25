# Candidate Generation Miss Taxonomy & Error Breakdown

**Benchmark Scope:** 10,000 Source-1 Entities | 34,511 True Ground-Truth Pairs
**Current Champion Candidate Recall:** 94.97% (32,774 / 34,511 retrieved)
**Total Missed Pairs Analyzed:** 1,737 (5.03% of ground truth)

## 1. Global Taxonomy Distribution

| Category | Description | Count | % of Misses | Primary Recovery Mechanism |
| :--- | :--- | :---: | :---: | :--- |
| **B TRANSLITERATION MISMATCH** | Complex Indic script or irregular phonetic spelling in native script. | 691 | 39.8% | Pass F: Transliteration backoff + script n-grams |
| **E ADDRESS REPRESENTATION MISMATCH** | Address phrasing variation, landmark, or street name abbreviation divergence. | 529 | 30.5% | Pass B & C: Structured address component & hierarchical retrieval |
| **D URL DOMAIN MISMATCH** | Domain vs company name discrepancy (e.g. `saffronfinancers.com` vs `Saffron Financers LLP`). | 174 | 10.0% | Pass A: Domain core & web-stem extraction |
| **H MISSING ADDRESS** | One entity has empty address, preventing composite address blocking. | 144 | 8.3% | Pass D: Rare-token backoff on distinctive name words |
| **F NUMERIC ADDRESS MISMATCH** | Address phrasing variation, landmark, or street name abbreviation divergence. | 98 | 5.6% | Pass B & C: Structured address component & hierarchical retrieval |
| **A NAME REPRESENTATION MISMATCH** | Name variation, legal form divergence, or minor typo. | 48 | 2.8% | Pass D & E: Informative token & n-gram backoff |
| **C DBA ALIAS MISMATCH** | Unlinked DBA, trade brand name, or parent subsidiary variation. | 31 | 1.8% | Pass A: Acronym / initialism matching |
| **J OTHER** | Extreme multi-field divergence across both name and address. | 15 | 0.9% | Pass D & E: Informative token & n-gram backoff |
| **G GEOGRAPHIC GRANULARITY MISMATCH** | One address has fine-grained plot/street, other only has city/state. | 7 | 0.4% | Pass B & C: Structured address component & hierarchical retrieval |


## 2. Representative Concrete Missed Cases

### B TRANSLITERATION MISMATCH (691 misses)

**Case 1:** (Name Sim: 13.33333333333333%, Addr Sim: 95.59748427672956%, S1 Nums: ['5', '2', '6', '101'], C Nums: ['5', '2', '6', '101'])
- **S1:** `Great Impex Private Limited` | `6-2-101/5/C, Telangana, Hyderabad, Secunderabad, Lane Beside Centralview Apt New Bhoiguda`
- **Cand:** `గ్రేట్ ఇంపెక్స్ ప్రైవేట్ లిమిటెడ్` | `6-2-101/5/C, Lane Beside Centralview Apt New Bhoiguda, Secunderabad, Hyderabad, TG`

**Case 2:** (Name Sim: 11.32075471698113%, Addr Sim: 47.05882352941176%, S1 Nums: ['10'], C Nums: [])
- **S1:** `Prime Projects Private Limited` | `S/O M Vasanthan, No.10 West Karikalan Street, Adambakkam, Chennai, Tamil Nadu`
- **Cand:** `பிரைம் புராஜெக்ட்ஸ் பிரைவேட் லிமிடெட்` | `S/O M VASANTHAN, KANCHEEPURAM, தமிழ்நாடு`

**Case 3:** (Name Sim: 11.940298507462687%, Addr Sim: 86.82926829268293%, S1 Nums: [], C Nums: ['662'])
- **S1:** `Big Aditya Technologies Private Limited` | `Fn0104 A Meenaxi Apt, Mumbai City, Goregaon (E), Maharashtra, Owners Assoc Gokuldam Near Gokuldham Mandir`
- **Cand:** `बिग आदित्य टेक्नोलॉजीज प्राइवेट लिमिटेड` | `H.NO 662- FN0104 A MEENAXI APT, OWNERS ASSOC GOKULDAM NEAR GOKULDHAM MANDIR, GOREGAON (E), MUMBAI CITY, महाराष्ट्र`

### E ADDRESS REPRESENTATION MISMATCH (529 misses)

**Case 1:** (Name Sim: 9.523809523809524%, Addr Sim: 87.07482993197279%, S1 Nums: [], C Nums: ['264'])
- **S1:** `Shakti Agro Limited` | `C/O Gurnav Singh Saluja, Beside Zudio, Kultapara, Sadar, Sambalpur, Orissa`
- **Cand:** `Wexveo` | `Block F-264- C/o Gurnav Singh Saluja, Beside Zudio, Kultapara, Sadar, Sambalpur, OD`

**Case 2:** (Name Sim: 66.66666666666667%, Addr Sim: 60.67415730337078%, S1 Nums: ['31'], C Nums: [])
- **S1:** `Jan India) Private Limited` | `Mumbai City, Maharashtra, Mumbai, 31 Sahakar Building, B Road, Churchgate`
- **Cand:** `Jan lndia) Private` | `Sahakar Building, Mumbai City, MH`

**Case 3:** (Name Sim: 68.57142857142857%, Addr Sim: 78.94736842105263%, S1 Nums: ['2200', '2696'], C Nums: ['2200', '2696'])
- **S1:** `PUS Horizon Paper` | `2696 2200, West Haven, UT`
- **Cand:** `pushorizonpapercom` | `UT, #2696 2200, OGDEN`

### D URL DOMAIN MISMATCH (174 misses)

**Case 1:** (Name Sim: 43.90243902439024%, Addr Sim: 77.77777777777779%, S1 Nums: [], C Nums: [])
- **S1:** `Saffron Financers LLP` | `47Akumaresa Puram Tiruchengode, Namakkal District, Salem, Tamil Nadu`
- **Cand:** `saffronfinancers.com` | `TN, Namakkal District, <NULL>, H.no 47Akumaresa Purma Tiruchengode`

**Case 2:** (Name Sim: 51.42857142857142%, Addr Sim: 35.08771929824561%, S1 Nums: ['9', '7', '299'], C Nums: [])
- **S1:** `House Food Pvt Ltd` | `3Rd Floor, S. No. 9/7, F.P. No. 299, Opp. Roopali Hotel, F.C. Road, Ganeshwad, I, Pune, Maharashtra`
- **Cand:** `Sri housefood.com` | `No 3Rd Floor, Poona, Erandwane, महाराष्ट्र`

**Case 3:** (Name Sim: 37.83783783783784%, Addr Sim: 92.21556886227545%, S1 Nums: ['12'], C Nums: ['12'])
- **S1:** `Vision Tech Private Limited` | `Plot No. 12C, North West, Delhi, Mezzanine Floor, Block C2, Pragati Market, Ashok Vihar, P, Hase Ii, Delhi`
- **Cand:** `M/s visi0ntech.com` | `PLOT NO. 12C  , MEZZANINE FLOOR, BLOCK C2, PRAGATI MARKET, ASHOK VIHAR, P, HASE II, DELHI, दिल्ली`

### H MISSING ADDRESS (144 misses)

**Case 1:** (Name Sim: 92.6829268292683%, Addr Sim: 0%, S1 Nums: ['12'], C Nums: [])
- **S1:** `LHB Interiors Pvt Ltd` | `No.12, Prem Nagar, Sewanagar, Gwalior, Madhya Pradesh`
- **Cand:** `LHB Integrir Pvt Ltd` | ``

**Case 2:** (Name Sim: 35.71428571428571%, Addr Sim: 0%, S1 Nums: ['3', '38', '7'], C Nums: [])
- **S1:** `Sai Construction` | `5Th Floor, Bbr Avenue, Plot No 38, C Block 7/3, Brindavanamu Colony, Kokapet, Rajendranagar, K.V.Rangareddy, Telangana`
- **Cand:** `Sai  Services` | ``

**Case 3:** (Name Sim: 91.66666666666666%, Addr Sim: 0%, S1 Nums: ['3', '26', '17', '6', '563'], C Nums: [])
- **S1:** `CQJ Consultancy Private Limited` | `6-3-563/26/A, Flat No.17, Somavarapu Heights Hilltop Colony, Erramanzil, Hyderabad, Telangana`
- **Cand:** `CQJ Consultatoscy Private Limited` | ``

### F NUMERIC ADDRESS MISMATCH (98 misses)

**Case 1:** (Name Sim: 60.0%, Addr Sim: 70.76923076923076%, S1 Nums: ['644'], C Nums: ['64', '1667'])
- **S1:** `Serrano Fresh Top, LLC` | `644 End Drive, Brookhaven, NY`
- **Cand:** `serranofreshtop.com` | `64 End Dr, PO Box 1667, Brookhaven, New York`

**Case 2:** (Name Sim: 16.666666666666664%, Addr Sim: 94.5945945945946%, S1 Nums: ['108'], C Nums: ['116'])
- **S1:** `Anny Adams Bridge` | `WA, College Place, 108 Mountain View Drive`
- **Cand:** `Wexlyra` | `116 MOUNTAIN VIEW DRIVE, COLLEGE PLACE, WA`

**Case 3:** (Name Sim: 93.61702127659575%, Addr Sim: 69.4915254237288%, S1 Nums: ['2', '344', '19'], C Nums: ['34'])
- **S1:** `Classic Global Private Limited` | `Sector 19, Pocket-2, Delhi, New Delhi, Flat No..344 Greenview Apartments, South West Delhi`
- **Cand:** `Dr Classic Global Private  Limited` | `Flat No..34 Greenview Apartmets, West Delhi, DL`

### A NAME REPRESENTATION MISMATCH (48 misses)

**Case 1:** (Name Sim: 83.33333333333334%, Addr Sim: 44.26229508196722%, S1 Nums: [], C Nums: [])
- **S1:** `Right & Co Corp` | `Shivaji Nagar, Behind Durgaweigh Bridge Near Saibaba Temple, Manpada (Chitalsar), Thane, Maharashtra`
- **Cand:** `Rrgieth & Co Corp` | `Thane, Shivaji Ngaar, MH, North Goa`

**Case 2:** (Name Sim: 90.9090909090909%, Addr Sim: 41.379310344827594%, S1 Nums: ['69'], C Nums: [])
- **S1:** `Southern Enterprises Limited` | `2Nd Floor, Plot-69, Anand Vihar, Bhulabhai Desai Road, Cumballa Hill, Mumbai, Mumbai City, Maharashtra`
- **Cand:** `Southern Énterprises` | `Mumbai(west), Mumbai, 2Nd- Floor, MH`

**Case 3:** (Name Sim: 66.66666666666667%, Addr Sim: 38.55421686746988%, S1 Nums: ['84', '9', '210', '13'], C Nums: ['210'])
- **S1:** `QE Trading Private Limited` | `Flat No. 210, S No. 84/9 To 13 Gajanan Darshan, Khed, Satara, Maharashtra`
- **Cand:** `#qetrading` | `No 210, Satara, MH`

### C DBA ALIAS MISMATCH (31 misses)

**Case 1:** (Name Sim: 67.3913043478261%, Addr Sim: 100.0%, S1 Nums: [], C Nums: [])
- **S1:** `Big Aditya Technologies Private Limited` | `Fn0104 A Meenaxi Apt, Mumbai City, Goregaon (E), Maharashtra, Owners Assoc Gokuldam Near Gokuldham Mandir`
- **Cand:** `Evosol Labs formerly known as Big Aditya Technologies Private Limited` | `Fn0104 A Meenaxi Apt, Owners Assoc Gokuldam Near Gokuldham Mandir, Goregaon (E), Mumbai City, Maharashtra`

**Case 2:** (Name Sim: 94.44444444444444%, Addr Sim: 100.0%, S1 Nums: ['100', '7175'], C Nums: ['100', '7175'])
- **S1:** `Womens Health Group` | `7175 Highway 100, Bon Aqua, TN`
- **Cand:** `Womens  Halh Group` | `7175 HIGHWAY 100, BON AQUA, TN`

**Case 3:** (Name Sim: 16.666666666666664%, Addr Sim: 76.76767676767678%, S1 Nums: ['157', '4'], C Nums: ['3', '157'])
- **S1:** `Tree Industries Group Private Limited` | `4/157, Lucknow, Uttar Pradesh, Vipul Khand Gomti Nagar, Lucknow`
- **Cand:** `Jaxorbi` | `3/157, Vipul Khand Gomti Nagar, Lucknow, UP`

### J OTHER (15 misses)

**Case 1:** (Name Sim: 22.22222222222222%, Addr Sim: 43.103448275862064%, S1 Nums: ['207'], C Nums: ['20'])
- **S1:** `White Infrastructure Pvt Ltd` | `Office-No-207, Marathon Monte Plaza, Madan Mohan Mamumbai, Mumbai, Mumbai City, Maharashtra`
- **Cand:** `Arianovi` | `Ofifce-no-20, Mumbai, Bhandup, MH`

**Case 2:** (Name Sim: 46.51162790697675%, Addr Sim: 42.85714285714286%, S1 Nums: ['95'], C Nums: ['6'])
- **S1:** `Victory Industries Private Limited` | `15Th Floor, Urmi Estate, Tower A 95, Ganpatrao Kadam Marg, Lower Parel We, St, Mumbai, Mumbai City, Maharashtra`
- **Cand:** `@victoryindustries` | `No 15Th/6 Floor, Mumbai, Mumbai City, महाराष्ट्र`

**Case 3:** (Name Sim: 17.777777777777782%, Addr Sim: 45.09803921568627%, S1 Nums: ['1', '35', '4'], C Nums: ['1', '35', '4'])
- **S1:** `Tejtech Tradelinks Pvt Ltd` | `Sr.No.35/4/1/1, Flat No.-4, Sita Residency Ganara, Pune, Maharashtra`
- **Cand:** `Evovio - 5537801104` | `महाराष्ट्र, null, Pune, Sr.no.#35/4/1/1, Pune`

### G GEOGRAPHIC GRANULARITY MISMATCH (7 misses)

**Case 1:** (Name Sim: 24.390243902439025%, Addr Sim: 30.107526881720425%, S1 Nums: ['3', '77'], C Nums: ['3', '7'])
- **S1:** `Seabird (India) Projects-Lucknow` | `Lucknow, 3/77, Lucknow, Vipul Khand, Opp. Study Hall School Gomtinagar, Uttar Pradesh`
- **Cand:** `Xylonexbrix` | `3/7, Lucknow, UP`

**Case 2:** (Name Sim: 28.57142857142857%, Addr Sim: 47.19101123595506%, S1 Nums: ['26', '281', '55'], C Nums: ['26', '281'])
- **S1:** `Green College` | `Door No.26/281, Sadhoo Company Road, No.55 Kra, Ondenparamb, Kannur, Kerala`
- **Cand:** `Lyraonyx` | `26/281, Kannur, Keralam`

**Case 3:** (Name Sim: 25.806451612903224%, Addr Sim: 31.147540983606557%, S1 Nums: ['1', '313', '12'], C Nums: ['1', '313', '12'])
- **S1:** `Shiv Energy Private Limited` | `313/12/1, 2Nd Floor, Above Icici Bank Khun Khun Ji Road, Near Munnu Lal Dharmsh, Ala, Chowk, Lucknow, Uttar Pradesh`
- **Cand:** `M/s Fluxonyx` | `313/12/1, Lucknow, UP`

