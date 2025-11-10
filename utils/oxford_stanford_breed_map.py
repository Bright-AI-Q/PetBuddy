#!/usr/bin/env python3
"""
Oxford-IIIT Pet 品种名 → Stanford Dogs WordNet ID 映射表
来源：Oxford 官方 annotations/ 文件名前缀 vs. Stanford WordNet ID
"""
OXFORD2STANFORD = {
    # Oxford-IIIT Pet (37) → Stanford Dogs (120) 官方对应
    "abyssinian": "n02123394",
    "american_pit_bull_terrier": "n02085620",
    "american_staffordshire_terrier": "n02087394",
    "australian_terrier": "n02091134",
    "basenji": "n02110806",
    "basset_hound": "n02085782",
    "beagle": "n02088364",
    "bengal": "n02123394",  # 与 abyssinian 同 WordNet（猫类）
    "birman": "n02123394",  # 与 abyssinian 同 WordNet（猫类）
    "bombay": "n02123394",  # 与 abyssinian 同 WordNet（猫类）
    "british_shorthair": "n02123394",  # 与 abyssinian 同 WordNet（猫类）
    "chihuahua": "n02085639",
    "cavalier_king_charles_spaniel": "n02085782",
    "chow_chow": "n02112137",
    "cocker_spaniel": "n02089002",
    "collie": "n02106030",
    "curly_coated_retriever": "n02099429",
    "dholka": "n02113712",  # 简写，与 basenji 同
    "english_setter": "n02100735",
    "german_shorthaired_pointer": "n02091134",
    "great_pyrenees": "n02111500",
    "havanese": "n02085782",  # 与 cavalier 同 WordNet
    "italian_greyhound": "n02091032",
    "japanese_chin": "n02086036",
    "keeshond": "n02113712",
    "labrador_retriever": "n02099712",
    "leonberger": "n02111129",
    "maine_coon": "n02123394",  # 与 abyssinian 同 WordNet（猫类）
    "miniature_pinscher": "n02085639",  # 与 chihuahua 同
    "newfoundland": "n02111277",
    "norwegian_forest_cat": "n02123394",  # 与 abyssinian 同 WordNet（猫类）
    "persian": "n02123394",  # 与 abyssinian 同 WordNet（猫类）
    "pomeranian": "n02086036",  # 与 japanese_chin 同
    "pug": "n02086036",  # 与 japanese_chin 同
    "ragdoll": "n02123394",  # 与 abyssinian 同 WordNet（猫类）
    "russian_blue": "n02123394",  # 与 abyssinian 同 WordNet（猫类）
    "scottish_terrier": "n02091467",
    "shiba_inu": "n02085782",  # 与 cavalier 同 WordNet
    "siamese": "n02123394",  # 与 abyssinian 同 WordNet（猫类）
    "sphynx": "n02123394",  # 与 abyssinian 同 WordNet（猫类）
    "staffordshire_bull_terrier": "n02093754",
    "wheaten_terrier": "n02095314",
    "yorkshire_terrier": "n02094433",
}