#!/usr/bin/env python3
"""
Training script for SabiYarn models with support for:
- All attention mechanisms (MHA, MLA, Differential Attention)
- MoE, Multi-Token Prediction, Layer Sharing
- Custom causal masking
- Auto-distributed training detection
"""

import os
import sys
import time
import math
from datetime import datetime
import json
import random
import string
import modal

project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

try:
    from dotenv import load_dotenv
    load_dotenv()
    ENV_FILE_LOADED = True
except ImportError:
    ENV_FILE_LOADED = False

# SabiYarn imports
from data import prepare


## The below parameter should be used for only testing
datasets_to_files = {
    # "Aletheia-ng/cosmopedia-100k": None,
    "Aletheia-ng/low_resource_languages_pretrain": [
        "akan-english_sentence-pairs_Akan_translation_batch_266312.parquet",
        "akan-english_sentence-pairs_Akan_translation_batch_266312.parquet",
        "english-ewe_sentence-pairs_English_translation_batch_16644.parquet",
        "english-ewe_sentence-pairs_English_translation_batch_16644.parquet",
        "english-ewe_sentence-pairs_English_translation_batch_16644.parquet",
        "english-ewe_sentence-pairs_English_translation_batch_800000.parquet",
        "english-ewe_sentence-pairs_English_translation_batch_800000.parquet",
        "english-ewe_sentence-pairs_English_translation_batch_800000.parquet",
        "english-fon_sentence-pairs_English_translation_batch_631368.parquet",
        "english-fon_sentence-pairs_English_translation_batch_631368.parquet",
        "english-fon_sentence-pairs_English_translation_batch_631368.parquet",
        "english-fon_sentence-pairs_English_translation_batch_800000.parquet",
        "english-fon_sentence-pairs_English_translation_batch_800000.parquet",
        "english-fon_sentence-pairs_English_translation_batch_800000.parquet",
        "english-fulah_sentence-pairs_English_translation_batch_622138.parquet",
        "english-fulah_sentence-pairs_English_translation_batch_622138.parquet",
        "english-fulah_sentence-pairs_English_translation_batch_622138.parquet",
        "english-fulah_sentence-pairs_English_translation_batch_622138.parquet",
        "english-fulah_sentence-pairs_English_translation_batch_800000.parquet",
        "english-fulah_sentence-pairs_English_translation_batch_800000.parquet",
        "english-fulah_sentence-pairs_English_translation_batch_800000.parquet",
        "english-fulah_sentence-pairs_English_translation_batch_800000.parquet",
        "english-hausa_sentence-pairs_English_translation_batch_757952.parquet",
        "english-hausa_sentence-pairs_English_translation_batch_757952.parquet",
        "english-hausa_sentence-pairs_English_translation_batch_757952.parquet",
        "english-hausa_sentence-pairs_English_translation_batch_800000.parquet",
        "english-hausa_sentence-pairs_English_translation_batch_800000.parquet",
        "english-hausa_sentence-pairs_English_translation_batch_800000.parquet",
        "english-igbo_sentence-pairs_English_translation_batch_220554.parquet",
        "english-igbo_sentence-pairs_English_translation_batch_220554.parquet",
        "english-igbo_sentence-pairs_English_translation_batch_220554.parquet",
        "english-igbo_sentence-pairs_English_translation_batch_800000.parquet",
        "english-igbo_sentence-pairs_English_translation_batch_800000.parquet",
        "english-igbo_sentence-pairs_English_translation_batch_800000.parquet",
        "english-twi_sentence-pairs_English_translation_batch_74878.parquet",
        "english-twi_sentence-pairs_English_translation_batch_74878.parquet",
        "english-twi_sentence-pairs_English_translation_batch_74878.parquet",
        "english-twi_sentence-pairs_English_translation_batch_74878.parquet",
        "english-twi_sentence-pairs_English_translation_batch_800000.parquet",
        "english-twi_sentence-pairs_English_translation_batch_800000.parquet",
        "english-twi_sentence-pairs_English_translation_batch_800000.parquet",
        "english-twi_sentence-pairs_English_translation_batch_800000.parquet",
        "english-yoruba_sentence-pairs_English_translation_batch_613512.parquet",
        "english-yoruba_sentence-pairs_English_translation_batch_613512.parquet",
        "english-yoruba_sentence-pairs_English_translation_batch_613512.parquet",
        "english-yoruba_sentence-pairs_English_translation_batch_800000.parquet",
        "english-yoruba_sentence-pairs_English_translation_batch_800000.parquet",
        "english-yoruba_sentence-pairs_English_translation_batch_800000.parquet",
        "twi-english-parallel-synthetic-50m_twi_translation_batch_800000.parquet",
        "twi-english-parallel-synthetic-50m_twi_translation_batch_800000.parquet",
        "twi-english-parallel-synthetic-50m_twi_translation_batch_800000.parquet",
        "twi-english-parallel-synthetic-50m_twi_translation_batch_800000.parquet",
        "twi-fante-sentences-parts-of-speech-pos-10m_Twi_monolingual_batch_287450.parquet",
        "twi-fante-sentences-parts-of-speech-pos-10m_Twi_monolingual_batch_287450.parquet",
        "twi-fante-sentences-parts-of-speech-pos-10m_Twi_monolingual_batch_800000.parquet",
        "twi-fante-sentences-parts-of-speech-pos-10m_Twi_monolingual_batch_800000.parquet",
        "twi-speech-text-parallel-multispeaker_Twi_monolingual_batch_21138.parquet",
        "twi-speech-text-parallel-multispeaker_Twi_monolingual_batch_21138.parquet",
        "twi_multispeaker_audio_transcribed_Twi_monolingual_batch_28063.parquet",
        "twi_multispeaker_audio_transcribed_Twi_monolingual_batch_28063.parquet",
        "twi_multispeaker_audio_transcribed_Twi_monolingual_batch_3187.parquet",
        "twi_multispeaker_audio_transcribed_Twi_monolingual_batch_3187.parquet"
            ],
    "Aletheia-ng/bloomberg-news-articles-pretraining-dataset": None
    }

if __name__ == "__main__":
    # For testing purposes only
    print("Downloading and preprocessing datasets...")
    prepare.run(datasets_to_files.keys(), datasets_to_files, os.cpu_count(), n_samples=None, seed=42,
                hash_algo="sha256", registry_cache="global_hash_registry.lmdb", map_size_gb=50)
    print("Done!")