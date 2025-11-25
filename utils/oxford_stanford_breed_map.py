#!/usr/bin/env python3
"""
Project: PetBuddy
Author: Bright Wang
File: oxford_stanford_breed_map.py
====================================
Oxford to Stanford Breed Mapping
Purpose:
- Map Oxford-IIIT breed names to Stanford Dogs IDs
- Only includes dog breeds (cats removed)
- Used for dataset merging and alignment
"""

OXFORD2STANFORD = {
    # Dog breed mappings
    "american_pit_bull_terrier": "n02085620",
    "american_staffordshire_terrier": "n02087394",
    "australian_terrier": "n02091134",
    "basenji": "n02110806",
    "basset_hound": "n02085782",
    "beagle": "n02088364",
    "chihuahua": "n02085639",
    "cavalier_king_charles_spaniel": "n02085782",
    "chow_chow": "n02112137",
    "cocker_spaniel": "n02089002",
    "collie": "n02106030",
    "curly_coated_retriever": "n02099429",
    "dholka": "n02110806",
    "english_setter": "n02100735",
    "german_shorthaired_pointer": "n02080739",
    "great_pyrenees": "n02111500",
    "havanese": "n02085782",
    "italian_greyhound": "n02091032",
    "japanese_chin": "n02086036",
    "keeshond": "n02113712",
    "labrador_retriever": "n02099712",
    "leonberger": "n02111129",
    "miniature_pinscher": "n02085639",
    "newfoundland": "n02111277",
    "pomeranian": "n02086036",
    "pug": "n02086036",
    "scottish_terrier": "n02091467",
    "shiba_inu": "n02085782",
    "staffordshire_bull_terrier": "n02093754",
    "wheaten_terrier": "n02095314",
    "yorkshire_terrier": "n02094433",
}
