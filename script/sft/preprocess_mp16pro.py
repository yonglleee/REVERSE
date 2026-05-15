#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_mp16pro.py
=====================
Preprocess MP16-Pro dataset into verl SFT parquet format.

Reads geotagged image paths from the extracted images directory
(images/XX/YY/XX_YY_nnnnnnn.jpg), builds geo-location QA pairs in
verl SFT messages format, and saves to train.parquet + test.parquet.

Image bytes are NOT embedded in the parquet -- only the absolute file
path is stored. This keeps parquet files small and allows versions to
share the same image files on disk.

Supported versions (--version, comma-separated or "all"):
  l0  --  answer format: <answer>Latitude, Longitude</answer>
  l2  --  answer format: <answer>Country, City, Latitude, Longitude</answer>           (= v1)
  l3  --  answer format: <answer>Country, Region, City, Latitude, Longitude</answer>
  l4  --  answer format: <answer>Country, Region, City, Neighbourhood, Latitude, Longitude</answer>
  v1  --  alias for l2 (backward compat)
  v2  --  answer format: <answer>Country, City, Neighbourhood, Latitude, Longitude</answer> (legacy)

Output directory layout:
  <output-dir>/<version>/train.parquet
  <output-dir>/<version>/test.parquet

When any of l3/l4 is requested, the script automatically filters to rows
where country, region, city, and neighbourhood are all non-empty (so the
same 200k sample is used across all granularity experiments).

Output parquet schema (messages column):
  list<struct<role: string, content: list<struct<type: string, image: string, text: string>>>>

Usage
-----
# l2, test-ratio 0.025, output to v2
cd /mnt/sh/mmvision/home/jonahli/projects/tusou/script/sft
python preprocess_mp16pro.py \
    --version l2 \
    --test-ratio 0.025 \
    --output-dir /mnt/sh/mmvision/home/jonahli/data/MP16-Pro/sft/v2 \
    --max-samples 0 --workers 64

# All 4 granularity levels, 200k samples, 64 workers
python preprocess_mp16pro.py --version all --max-samples 200000 --workers 64 \\
    --csv /mnt/sh/mmvision/home/jonahli/data/MP16-Pro/metadata/MP16_Pro_fixed_clean.csv

# Only l3
python preprocess_mp16pro.py --version l3 --max-samples 200000

# Quick test with 1000 samples
python preprocess_mp16pro.py --version all --max-samples 1000 --output-dir /tmp/mp16pro_sft_test

# Custom paths
python preprocess_mp16pro.py \\
    --csv        /mnt/sh/mmvision/home/jonahli/data/MP16-Pro/metadata/MP16_Pro_fixed_clean.csv \\
    --images-dir /mnt/sh/mmvision/home/jonahli/data/MP16-Pro/images \\
    --output-dir /mnt/sh/mmvision/home/jonahli/data/MP16-Pro/sft_granularity_parquet \\
    --version all --max-samples 200000 --workers 64 --test-ratio 0.02
"""

import argparse
import json
import logging
import os
import random
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CSV        = "/mnt/sh/mmvision/home/jonahli/data/MP16-Pro/metadata/MP16_Pro_fixed_clean.csv"
DEFAULT_IMAGES_DIR = "/mnt/sh/mmvision/home/jonahli/data/MP16-Pro/images"
DEFAULT_OUTPUT_DIR = "/mnt/sh/mmvision/home/jonahli/data/MP16-Pro/sft_granularity_parquet"

# ---------------------------------------------------------------------------
# PyArrow schema (verl-compatible format)
# ---------------------------------------------------------------------------
# messages: content is a plain string with <image> placeholder
# images: separate column with list of {image: path} dicts

MESSAGE_TYPE = pa.struct([
    pa.field("role",    pa.string()),
    pa.field("content", pa.string()),
])

IMAGE_ITEM_TYPE = pa.struct([
    pa.field("image", pa.string()),
])

PARQUET_SCHEMA = pa.schema([
    pa.field("messages", pa.list_(MESSAGE_TYPE)),
    pa.field("images",   pa.list_(IMAGE_ITEM_TYPE)),
])

# ---------------------------------------------------------------------------
# Question pools
# ---------------------------------------------------------------------------

# ── l0: direct coordinates, no place names ──────────────────────────────────
QUESTION_POOL_L0 = [
    # Direct / concise
    "Where was this photo taken? Answer as: Latitude, Longitude.",
    "Where is this? Latitude, Longitude.",
    "Geolocate this image. Answer: Latitude, Longitude.",
    "What are the GPS coordinates of this location? Provide: Latitude, Longitude.",
    "Give the coordinates of the place in this photo. Format: Latitude, Longitude.",

    # Instruction-style
    "Identify the GPS coordinates of this image. Format: Latitude, Longitude.",
    "Provide the geographic coordinates of this photo in the format: Latitude, Longitude.",
    "Output the coordinates of this image as: Latitude, Longitude.",
    "Return the GPS location where this photo was taken. Format: Latitude, Longitude.",
    "Report the coordinates of the location shown in this image. Format: Latitude, Longitude.",

    # Reasoning-cue style
    "Look at the visual clues in this photo -- architecture, signs, vegetation, terrain. What are the GPS coordinates? Answer: Latitude, Longitude.",
    "Use the street scene, buildings, and environment to estimate the GPS coordinates of this photo. Answer: Latitude, Longitude.",
    "Examine the landscape, signage, and cultural markers. What are the coordinates of this location? Format: Latitude, Longitude.",
    "Based on the architecture and surroundings visible here, estimate the GPS coordinates. Answer: Latitude, Longitude.",
    "Study the visual context -- vegetation, road markings, building styles. What are the coordinates? Answer: Latitude, Longitude.",
    "Analyze the scene and estimate the GPS coordinates. Format: Latitude, Longitude.",
    "What geographic clues in this image help you estimate its coordinates? Provide: Latitude, Longitude.",
    "The scene contains visual hints about its location. What are the GPS coordinates? Format: Latitude, Longitude.",

    # Conversational / natural
    "Hey, do you know the coordinates of where this photo was taken? Give me: Latitude, Longitude.",
    "Can you estimate the GPS location of this scene? Reply with: Latitude, Longitude.",
    "I'm trying to figure out the coordinates of this photo. Can you help? Answer: Latitude, Longitude.",
    "Do you recognize this place? Give me the GPS coordinates.",
    "Where in the world is this? Format your answer as: Latitude, Longitude.",
    "Any idea where this was shot? Please give the coordinates: Latitude, Longitude.",
    "What coordinates do you think this is? Format: Latitude, Longitude.",

    # Task-framing style
    "Your task: determine the GPS coordinates of the scene in this image. Answer format: Latitude, Longitude.",
    "Given this image, predict the GPS coordinates where it was captured. Output: Latitude, Longitude.",
    "Estimate the GPS coordinates of this photo. Format: Latitude, Longitude.",
    "This is a geolocation task. Identify the coordinates of this photo. Output: Latitude, Longitude.",
    "Perform geographic localization on this image. Return the GPS coordinates: Latitude, Longitude.",

    # Specificity-oriented
    "What are the latitude and longitude of this location? Format: Latitude, Longitude.",
    "Give me the latitude first, then the longitude. Format: Latitude, Longitude.",
    "What are the exact GPS coordinates of this photo? Answer: Latitude, Longitude.",
    "Pin this image to a point on the map. Provide: Latitude, Longitude.",

    # Implicit-reasoning style
    "Imagine you are a geographer. What are the coordinates of this location? Answer: Latitude, Longitude.",
    "If you had to place this photo on a world map, what coordinates would you use? Format: Latitude, Longitude.",
    "A photo posted online without location data. Based on the scene, what are the GPS coordinates? Answer: Latitude, Longitude.",
    "Pretend you're playing GeoGuessr. What coordinates would you enter? Format: Latitude, Longitude.",
    "You are a location recognition AI. What are the GPS coordinates of this photo? Output: Latitude, Longitude.",

    # Minimal / telegraphic
    "GPS coordinates? (Latitude, Longitude)",
    "Coordinates? Answer: Latitude, Longitude.",
    "Photo GPS -- Latitude, Longitude.",
    "Identify coordinates: Latitude, Longitude.",

    # Multi-sentence elaborated
    "This image was captured at an unknown location. Using all visible contextual clues, estimate its GPS coordinates. Provide the answer as: Latitude, Longitude.",
    "Looking at this photograph, consider the environment, infrastructure, and any recognizable elements. What GPS coordinates does it correspond to? Format: Latitude, Longitude.",
    "The photo contains geographic information encoded in its visual elements. Decode it and report the GPS coordinates: Latitude, Longitude.",
    "Without any metadata, can you still estimate the GPS coordinates of this photo just from looking at it? Answer as: Latitude, Longitude.",
    "Use your knowledge of global geography and visual recognition to estimate the GPS coordinates. Return: Latitude, Longitude.",
    "What does the environment in this photo tell you about its GPS coordinates? Provide the answer as: Latitude, Longitude.",
]

# ── l2 / v1: Country + City + coordinates ───────────────────────────────────
QUESTION_POOL_L2 = [
    # Direct / concise
    "Where was this photo taken? Answer as: Country, City, Latitude, Longitude.",
    "Where is this? Country, City, Latitude, Longitude.",
    "Geolocate this image. Answer: Country, City, Latitude, Longitude.",
    "What location is this? Provide: Country, City, Latitude, Longitude.",
    "Name the place in this photo. Format: Country, City, Latitude, Longitude.",

    # Instruction-style
    "Identify the country, city, and GPS coordinates of this image. Format: Country, City, Latitude, Longitude.",
    "Provide the geographic location of this photo in the format: Country, City, Latitude, Longitude.",
    "Output the location of this image as: Country, City, Latitude, Longitude.",
    "Return the location where this photo was taken. Format: Country, City, Latitude, Longitude.",
    "Report the location shown in this image. Format: Country, City, Latitude, Longitude.",

    # Reasoning-cue style
    "Look at the visual clues in this photo -- architecture, signs, vegetation, terrain. Where was it taken? Answer: Country, City, Latitude, Longitude.",
    "Use the street scene, buildings, and environment to determine where this photo was shot. Answer: Country, City, Latitude, Longitude.",
    "Examine the landscape, signage, and cultural markers. What location does this photo show? Format: Country, City, Latitude, Longitude.",
    "Based on the architecture and surroundings visible here, identify the location. Answer: Country, City, Latitude, Longitude.",
    "Study the visual context -- vegetation, road markings, building styles. Where is this? Answer: Country, City, Latitude, Longitude.",
    "Analyze the scene: what country and city is this, and what are the GPS coordinates? Format: Country, City, Latitude, Longitude.",
    "What geographic clues in this image reveal where it was taken? Provide: Country, City, Latitude, Longitude.",
    "The scene contains visual hints about its location. What are the country, city, and coordinates? Format: Country, City, Latitude, Longitude.",

    # Conversational / natural
    "Hey, do you know where this photo is from? Give me: Country, City, Latitude, Longitude.",
    "Can you tell what city this is? Reply with: Country, City, Latitude, Longitude.",
    "I'm trying to figure out where this photo was taken. Can you help? Answer: Country, City, Latitude, Longitude.",
    "Do you recognize this place? Tell me the country, city, and coordinates.",
    "Where in the world is this? Format your answer as: Country, City, Latitude, Longitude.",
    "Any idea where this was shot? Please give: Country, City, Latitude, Longitude.",
    "What city do you think this is? Include the country and GPS coordinates. Format: Country, City, Latitude, Longitude.",

    # Task-framing style
    "Your task: determine the geographic location of the scene in this image. Answer format: Country, City, Latitude, Longitude.",
    "Given this image, predict the location it was captured at. Output: Country, City, Latitude, Longitude.",
    "Estimate the country, city, and GPS coordinates of this photo. Format: Country, City, Latitude, Longitude.",
    "This is a geolocation task. Identify where this photo was taken. Output: Country, City, Latitude, Longitude.",
    "Perform geographic localization on this image. Return: Country, City, Latitude, Longitude.",

    # Specificity-oriented
    "What country and city is visible in this image? Also give the latitude and longitude. Format: Country, City, Latitude, Longitude.",
    "Identify the country first, then the city, then the GPS coordinates. Format: Country, City, Latitude, Longitude.",
    "Which country does this scene belong to, and what city? What are the coordinates? Answer: Country, City, Latitude, Longitude.",
    "Give me the exact country, city, and GPS location of this photo. Format: Country, City, Latitude, Longitude.",
    "Pin this image to a specific city on the map. Provide: Country, City, Latitude, Longitude.",

    # Implicit-reasoning style
    "Imagine you are a geographer. Where was this photo taken? Answer: Country, City, Latitude, Longitude.",
    "If you had to place this photo on a world map, where would it go? Format: Country, City, Latitude, Longitude.",
    "A photo posted online without location data. Based on the scene, where is it from? Answer: Country, City, Latitude, Longitude.",
    "Pretend you're playing GeoGuessr. Where is this? Format: Country, City, Latitude, Longitude.",
    "You are a location recognition AI. Where was this photo taken? Output: Country, City, Latitude, Longitude.",

    # Minimal / telegraphic
    "Location? (Country, City, Latitude, Longitude)",
    "Where? Answer: Country, City, Latitude, Longitude.",
    "Photo location -- Country, City, Latitude, Longitude.",
    "Identify: Country, City, Latitude, Longitude.",

    # Multi-sentence elaborated
    "This image was captured at an unknown location. Using all visible contextual clues, determine where it was taken and provide the answer as: Country, City, Latitude, Longitude.",
    "Looking at this photograph, consider the environment, infrastructure, and any recognizable elements. What location does it depict? Format: Country, City, Latitude, Longitude.",
    "The photo contains geographic information encoded in its visual elements. Decode it and report: Country, City, Latitude, Longitude.",
    "Without any metadata, can you still identify where this photo was taken just from looking at it? Answer as: Country, City, Latitude, Longitude.",
    "Use your knowledge of global geography and visual recognition to identify this location. Return: Country, City, Latitude, Longitude.",
    "What does the environment in this photo tell you about its location? Provide the answer as: Country, City, Latitude, Longitude.",
]

# ── l3: Country + Region + City + coordinates ────────────────────────────────
QUESTION_POOL_L3 = [
    # Direct / concise
    "Where was this photo taken? Answer as: Country, Region, City, Latitude, Longitude.",
    "Where is this? Country, Region, City, Latitude, Longitude.",
    "Geolocate this image. Answer: Country, Region, City, Latitude, Longitude.",
    "What location is this? Provide: Country, Region, City, Latitude, Longitude.",
    "Name the place in this photo, including the region. Format: Country, Region, City, Latitude, Longitude.",

    # Instruction-style
    "Identify the country, region, city, and GPS coordinates of this image. Format: Country, Region, City, Latitude, Longitude.",
    "Provide the geographic location of this photo in the format: Country, Region, City, Latitude, Longitude.",
    "Output the location of this image as: Country, Region, City, Latitude, Longitude.",
    "Return the location where this photo was taken. Format: Country, Region, City, Latitude, Longitude.",
    "Report the location shown in this image, including the region/state. Format: Country, Region, City, Latitude, Longitude.",

    # Reasoning-cue style
    "Look at the visual clues in this photo -- architecture, signs, vegetation, terrain. Where was it taken? Answer: Country, Region, City, Latitude, Longitude.",
    "Use the street scene, buildings, and environment to determine where this photo was shot. Answer: Country, Region, City, Latitude, Longitude.",
    "Examine the landscape, signage, and cultural markers. What location does this photo show, including the region? Format: Country, Region, City, Latitude, Longitude.",
    "Based on the architecture and surroundings visible here, identify the location with its region. Answer: Country, Region, City, Latitude, Longitude.",
    "Study the visual context -- vegetation, road markings, building styles. Where is this, and which region? Answer: Country, Region, City, Latitude, Longitude.",
    "Analyze the scene: what country, region, and city is this, and what are the GPS coordinates? Format: Country, Region, City, Latitude, Longitude.",
    "What geographic clues in this image reveal the country, region, and city? Provide: Country, Region, City, Latitude, Longitude.",
    "The scene contains visual hints about its location. What are the country, region, city, and coordinates? Format: Country, Region, City, Latitude, Longitude.",

    # Conversational / natural
    "Hey, do you know where this photo is from, including the region? Give me: Country, Region, City, Latitude, Longitude.",
    "Can you tell what region and city this is in? Reply with: Country, Region, City, Latitude, Longitude.",
    "I'm trying to figure out where this photo was taken. Can you help? Answer: Country, Region, City, Latitude, Longitude.",
    "Do you recognize this place? Tell me the country, region, city, and coordinates.",
    "Where in the world is this, and which region? Format: Country, Region, City, Latitude, Longitude.",
    "Any idea where this was shot? Please give: Country, Region, City, Latitude, Longitude.",
    "What region and city do you think this is in? Include the country and GPS coordinates. Format: Country, Region, City, Latitude, Longitude.",

    # Task-framing style
    "Your task: determine the geographic location of the scene in this image, including the region. Answer format: Country, Region, City, Latitude, Longitude.",
    "Given this image, predict the location it was captured at, including the region. Output: Country, Region, City, Latitude, Longitude.",
    "Estimate the country, region, city, and GPS coordinates of this photo. Format: Country, Region, City, Latitude, Longitude.",
    "This is a geolocation task. Identify where this photo was taken, including the region. Output: Country, Region, City, Latitude, Longitude.",
    "Perform geographic localization on this image at the region level. Return: Country, Region, City, Latitude, Longitude.",

    # Specificity-oriented
    "What country, region, and city is visible in this image? Also give the latitude and longitude. Format: Country, Region, City, Latitude, Longitude.",
    "Identify the country first, then the region, then the city, then the GPS coordinates. Format: Country, Region, City, Latitude, Longitude.",
    "Which country and region does this scene belong to, and what city? What are the coordinates? Answer: Country, Region, City, Latitude, Longitude.",
    "Give me the exact country, region, city, and GPS location of this photo. Format: Country, Region, City, Latitude, Longitude.",
    "Pin this image to a specific city in its region on the map. Provide: Country, Region, City, Latitude, Longitude.",

    # Implicit-reasoning style
    "Imagine you are a geographer. Where was this photo taken, including the region? Answer: Country, Region, City, Latitude, Longitude.",
    "If you had to place this photo on a regional map, where would it go? Format: Country, Region, City, Latitude, Longitude.",
    "A photo posted online without location data. Based on the scene, where is it from, including the region? Answer: Country, Region, City, Latitude, Longitude.",
    "Pretend you're playing GeoGuessr. Where is this, and which region? Format: Country, Region, City, Latitude, Longitude.",
    "You are a location recognition AI. Where was this photo taken, including the region? Output: Country, Region, City, Latitude, Longitude.",

    # Minimal / telegraphic
    "Location? (Country, Region, City, Latitude, Longitude)",
    "Where? Answer: Country, Region, City, Latitude, Longitude.",
    "Photo location -- Country, Region, City, Latitude, Longitude.",
    "Identify: Country, Region, City, Latitude, Longitude.",

    # Multi-sentence elaborated
    "This image was captured at an unknown location. Using all visible contextual clues, determine where it was taken and provide the answer as: Country, Region, City, Latitude, Longitude.",
    "Looking at this photograph, consider the environment, infrastructure, and any recognizable elements. What location does it depict, including the region? Format: Country, Region, City, Latitude, Longitude.",
    "The photo contains geographic information encoded in its visual elements. Decode it and report: Country, Region, City, Latitude, Longitude.",
    "Without any metadata, can you still identify the region and city where this photo was taken just from looking at it? Answer as: Country, Region, City, Latitude, Longitude.",
    "Use your knowledge of global geography and visual recognition to identify this location, including the region. Return: Country, Region, City, Latitude, Longitude.",
    "What does the environment in this photo tell you about its location? Provide the answer as: Country, Region, City, Latitude, Longitude.",
]

# ── l4: Country + Region + City + Neighbourhood + coordinates ────────────────
QUESTION_POOL_L4 = [
    # Direct / concise
    "Where was this photo taken? Answer as: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Where is this? Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Geolocate this image. Answer: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "What location is this? Provide: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Name the place in this photo, including the neighbourhood. Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",

    # Instruction-style
    "Identify the country, region, city, neighbourhood, and GPS coordinates of this image. Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Provide the geographic location of this photo in the format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Output the location of this image as: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Return the location where this photo was taken. Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Report the precise location shown in this image, including the neighbourhood. Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",

    # Reasoning-cue style
    "Look at the visual clues in this photo -- architecture, signs, vegetation, terrain. Where was it taken? Answer: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Use the street scene, buildings, and environment to determine where this photo was shot. Answer: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Examine the landscape, signage, and cultural markers. What location does this photo show, down to the neighbourhood? Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Based on the architecture and surroundings visible here, identify the location precisely. Answer: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Study the visual context -- vegetation, road markings, building styles. Where is this, and which neighbourhood? Answer: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Analyze the scene: what country, region, city, and neighbourhood is this, and what are the GPS coordinates? Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "What geographic clues in this image reveal where it was taken, including the local neighbourhood? Provide: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "The scene contains visual hints about its precise location. What are the country, region, city, neighbourhood, and coordinates? Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",

    # Conversational / natural
    "Hey, do you know where this photo is from, down to the neighbourhood? Give me: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Can you tell what neighbourhood this is in? Reply with: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "I'm trying to figure out exactly where this photo was taken. Can you help? Answer: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Do you recognize this place? Tell me the country, region, city, neighbourhood, and coordinates.",
    "Where in the world is this, and which part of the city? Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Any idea where this was shot? Please give: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "What neighbourhood do you think this is in? Include the country, region, city, and GPS coordinates. Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",

    # Task-framing style
    "Your task: determine the precise geographic location of the scene in this image, including neighbourhood. Answer format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Given this image, predict the location it was captured at, down to the neighbourhood level. Output: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Estimate the country, region, city, neighbourhood, and GPS coordinates of this photo. Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "This is a fine-grained geolocation task. Identify where this photo was taken, including the neighbourhood. Output: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Perform geographic localization on this image at the neighbourhood level. Return: Country, Region, City, Neighbourhood, Latitude, Longitude.",

    # Specificity-oriented
    "What country, region, city, and neighbourhood is visible in this image? Also give the latitude and longitude. Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Identify the country first, then the region, then the city, then the neighbourhood, then the GPS coordinates. Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Which country and region does this scene belong to, and what city and neighbourhood? What are the coordinates? Answer: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Give me the exact country, region, city, neighbourhood, and GPS location of this photo. Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Pin this image to a specific neighbourhood on the map. Provide: Country, Region, City, Neighbourhood, Latitude, Longitude.",

    # Implicit-reasoning style
    "Imagine you are a geographer. Where was this photo taken, and which neighbourhood? Answer: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "If you had to place this photo on a detailed city map, where would it go? Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "A photo posted online without location data. Based on the scene, where is it from, including the neighbourhood? Answer: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Pretend you're playing GeoGuessr at maximum zoom. Where is this, and which neighbourhood? Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "You are a location recognition AI. Where was this photo taken, down to the neighbourhood level? Output: Country, Region, City, Neighbourhood, Latitude, Longitude.",

    # Minimal / telegraphic
    "Location? (Country, Region, City, Neighbourhood, Latitude, Longitude)",
    "Where? Answer: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Photo location -- Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Identify: Country, Region, City, Neighbourhood, Latitude, Longitude.",

    # Multi-sentence elaborated
    "This image was captured at an unknown location. Using all visible contextual clues, determine where it was taken and provide the answer as: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Looking at this photograph, consider the environment, infrastructure, and any recognizable elements. What location does it depict, including the neighbourhood? Format: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "The photo contains geographic information encoded in its visual elements. Decode it and report: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Without any metadata, can you still identify the neighbourhood where this photo was taken just from looking at it? Answer as: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "Use your knowledge of global geography and visual recognition to identify this location at the neighbourhood level. Return: Country, Region, City, Neighbourhood, Latitude, Longitude.",
    "What does the environment in this photo tell you about its precise location? Provide the answer as: Country, Region, City, Neighbourhood, Latitude, Longitude.",
]

# ── v2 (legacy): Country + City + Neighbourhood + coordinates ────────────────
QUESTION_POOL_V2 = [
    # Direct / concise
    "Where was this photo taken? Answer as: Country, City, Neighbourhood, Latitude, Longitude.",
    "Where is this? Country, City, Neighbourhood, Latitude, Longitude.",
    "Geolocate this image. Answer: Country, City, Neighbourhood, Latitude, Longitude.",
    "What location is this? Provide: Country, City, Neighbourhood, Latitude, Longitude.",
    "Name the place in this photo, including the neighbourhood. Format: Country, City, Neighbourhood, Latitude, Longitude.",

    # Instruction-style
    "Identify the country, city, neighbourhood, and GPS coordinates of this image. Format: Country, City, Neighbourhood, Latitude, Longitude.",
    "Provide the geographic location of this photo in the format: Country, City, Neighbourhood, Latitude, Longitude.",
    "Output the location of this image as: Country, City, Neighbourhood, Latitude, Longitude.",
    "Return the location where this photo was taken. Format: Country, City, Neighbourhood, Latitude, Longitude.",
    "Report the precise location shown in this image, including the neighbourhood. Format: Country, City, Neighbourhood, Latitude, Longitude.",

    # Reasoning-cue style
    "Look at the visual clues in this photo -- architecture, signs, vegetation, terrain. Where was it taken? Answer: Country, City, Neighbourhood, Latitude, Longitude.",
    "Use the street scene, buildings, and environment to determine where this photo was shot. Answer: Country, City, Neighbourhood, Latitude, Longitude.",
    "Examine the landscape, signage, and cultural markers. What location does this photo show, down to the neighbourhood? Format: Country, City, Neighbourhood, Latitude, Longitude.",
    "Based on the architecture and surroundings visible here, identify the location precisely. Answer: Country, City, Neighbourhood, Latitude, Longitude.",
    "Study the visual context -- vegetation, road markings, building styles. Where is this, and which neighbourhood? Answer: Country, City, Neighbourhood, Latitude, Longitude.",
    "Analyze the scene: what country, city, and neighbourhood is this, and what are the GPS coordinates? Format: Country, City, Neighbourhood, Latitude, Longitude.",
    "What geographic clues in this image reveal where it was taken, including the local neighbourhood? Provide: Country, City, Neighbourhood, Latitude, Longitude.",
    "The scene contains visual hints about its precise location. What are the country, city, neighbourhood, and coordinates? Format: Country, City, Neighbourhood, Latitude, Longitude.",

    # Conversational / natural
    "Hey, do you know where this photo is from, down to the neighbourhood? Give me: Country, City, Neighbourhood, Latitude, Longitude.",
    "Can you tell what neighbourhood this is in? Reply with: Country, City, Neighbourhood, Latitude, Longitude.",
    "I'm trying to figure out exactly where this photo was taken. Can you help? Answer: Country, City, Neighbourhood, Latitude, Longitude.",
    "Do you recognize this place? Tell me the country, city, neighbourhood, and coordinates.",
    "Where in the world is this, and which part of the city? Format: Country, City, Neighbourhood, Latitude, Longitude.",
    "Any idea where this was shot? Please give: Country, City, Neighbourhood, Latitude, Longitude.",
    "What neighbourhood do you think this is in? Include the country, city, and GPS coordinates. Format: Country, City, Neighbourhood, Latitude, Longitude.",

    # Task-framing style
    "Your task: determine the precise geographic location of the scene in this image, including neighbourhood. Answer format: Country, City, Neighbourhood, Latitude, Longitude.",
    "Given this image, predict the location it was captured at, down to the neighbourhood level. Output: Country, City, Neighbourhood, Latitude, Longitude.",
    "Estimate the country, city, neighbourhood, and GPS coordinates of this photo. Format: Country, City, Neighbourhood, Latitude, Longitude.",
    "This is a fine-grained geolocation task. Identify where this photo was taken, including the neighbourhood. Output: Country, City, Neighbourhood, Latitude, Longitude.",
    "Perform geographic localization on this image at the neighbourhood level. Return: Country, City, Neighbourhood, Latitude, Longitude.",

    # Specificity-oriented
    "What country, city, and neighbourhood is visible in this image? Also give the latitude and longitude. Format: Country, City, Neighbourhood, Latitude, Longitude.",
    "Identify the country first, then the city, then the neighbourhood, then the GPS coordinates. Format: Country, City, Neighbourhood, Latitude, Longitude.",
    "Which country does this scene belong to, and what city and neighbourhood? What are the coordinates? Answer: Country, City, Neighbourhood, Latitude, Longitude.",
    "Give me the exact country, city, neighbourhood, and GPS location of this photo. Format: Country, City, Neighbourhood, Latitude, Longitude.",
    "Pin this image to a specific neighbourhood on the map. Provide: Country, City, Neighbourhood, Latitude, Longitude.",

    # Implicit-reasoning style
    "Imagine you are a geographer. Where was this photo taken, and which neighbourhood? Answer: Country, City, Neighbourhood, Latitude, Longitude.",
    "If you had to place this photo on a detailed city map, where would it go? Format: Country, City, Neighbourhood, Latitude, Longitude.",
    "A photo posted online without location data. Based on the scene, where is it from, including the neighbourhood? Answer: Country, City, Neighbourhood, Latitude, Longitude.",
    "Pretend you're playing GeoGuessr at maximum zoom. Where is this, and which neighbourhood? Format: Country, City, Neighbourhood, Latitude, Longitude.",
    "You are a location recognition AI. Where was this photo taken, down to the neighbourhood level? Output: Country, City, Neighbourhood, Latitude, Longitude.",

    # Minimal / telegraphic
    "Location? (Country, City, Neighbourhood, Latitude, Longitude)",
    "Where? Answer: Country, City, Neighbourhood, Latitude, Longitude.",
    "Photo location -- Country, City, Neighbourhood, Latitude, Longitude.",
    "Identify: Country, City, Neighbourhood, Latitude, Longitude.",

    # Multi-sentence elaborated
    "This image was captured at an unknown location. Using all visible contextual clues, determine where it was taken and provide the answer as: Country, City, Neighbourhood, Latitude, Longitude.",
    "Looking at this photograph, consider the environment, infrastructure, and any recognizable elements. What location does it depict, including the neighbourhood? Format: Country, City, Neighbourhood, Latitude, Longitude.",
    "The photo contains geographic information encoded in its visual elements. Decode it and report: Country, City, Neighbourhood, Latitude, Longitude.",
    "Without any metadata, can you still identify the neighbourhood where this photo was taken just from looking at it? Answer as: Country, City, Neighbourhood, Latitude, Longitude.",
    "Use your knowledge of global geography and visual recognition to identify this location at the neighbourhood level. Return: Country, City, Neighbourhood, Latitude, Longitude.",
    "What does the environment in this photo tell you about its precise location? Provide the answer as: Country, City, Neighbourhood, Latitude, Longitude.",
]

VERSION_CONFIGS = {
    # New granularity levels
    "l0": {
        "system_prompt": (
            "You are a geolocation expert. "
            "Given an image, analyze the visual cues such as architecture, vegetation, "
            "road signs, landscape, and cultural elements to determine the GPS coordinates. "
            "The final answer MUST BE enclosed in <answer></answer> tags with the format: "
            "Latitude, Longitude. "
            "e.g. <answer>40.9606, 9.5873</answer>"
        ),
        "question_pool": QUESTION_POOL_L0,
    },
    "l2": {
        "system_prompt": (
            "You are a geolocation expert. "
            "Given an image, analyze the visual cues such as architecture, vegetation, "
            "road signs, landscape, and cultural elements to determine the location. "
            "The final answer MUST BE enclosed in <answer></answer> tags with the format: "
            "Country, City, Latitude, Longitude. "
            "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
        ),
        "question_pool": QUESTION_POOL_L2,
    },
    "l3": {
        "system_prompt": (
            "You are a geolocation expert. "
            "Given an image, analyze the visual cues such as architecture, vegetation, "
            "road signs, landscape, and cultural elements to determine the location. "
            "The final answer MUST BE enclosed in <answer></answer> tags with the format: "
            "Country, Region, City, Latitude, Longitude. "
            "e.g. <answer>Italy, Sardinia, Golfo Arnaci, 40.9606, 9.5873</answer>"
        ),
        "question_pool": QUESTION_POOL_L3,
    },
    "l4": {
        "system_prompt": (
            "You are a geolocation expert. "
            "Given an image, analyze the visual cues such as architecture, vegetation, "
            "road signs, landscape, and cultural elements to determine the location. "
            "The final answer MUST BE enclosed in <answer></answer> tags with the format: "
            "Country, Region, City, Neighbourhood, Latitude, Longitude. "
            "e.g. <answer>Italy, Sardinia, Golfo Arnaci, Marina, 40.9606, 9.5873</answer>"
        ),
        "question_pool": QUESTION_POOL_L4,
    },
    # Legacy versions (backward compat)
    "v1": {
        "system_prompt": (
            "You are a geolocation expert. "
            "Given an image, analyze the visual cues such as architecture, vegetation, "
            "road signs, landscape, and cultural elements to determine the location. "
            "The final answer MUST BE enclosed in <answer></answer> tags with the format: "
            "Country, City, Latitude, Longitude. "
            "e.g. <answer>Italy, Golfo Arnaci, 40.9606, 9.5873</answer>"
        ),
        "question_pool": QUESTION_POOL_L2,
    },
    "v2": {
        "system_prompt": (
            "You are a geolocation expert. "
            "Given an image, analyze the visual cues such as architecture, vegetation, "
            "road signs, landscape, and cultural elements to determine the location. "
            "The final answer MUST BE enclosed in <answer></answer> tags with the format: "
            "Country, City, Neighbourhood, Latitude, Longitude. "
            "e.g. <answer>Italy, Golfo Arnaci, Marina, 40.9606, 9.5873</answer>"
        ),
        "question_pool": QUESTION_POOL_V2,
    },
}


# ---------------------------------------------------------------------------
# Answer formatting
# ---------------------------------------------------------------------------
def format_answer_l0(lat: float, lon: float) -> str:
    return "<answer>{:.4f}, {:.4f}</answer>".format(lat, lon)


def format_answer_l2(lat: float, lon: float, country: str, city: str) -> str:
    return "<answer>{}, {}, {:.4f}, {:.4f}</answer>".format(
        country.strip(), city.strip(), lat, lon)


def format_answer_l3(lat: float, lon: float, country: str, region: str, city: str) -> str:
    return "<answer>{}, {}, {}, {:.4f}, {:.4f}</answer>".format(
        country.strip(), region.strip(), city.strip(), lat, lon)


def format_answer_l4(lat: float, lon: float, country: str, region: str, city: str, neighbourhood: str) -> str:
    return "<answer>{}, {}, {}, {}, {:.4f}, {:.4f}</answer>".format(
        country.strip(), region.strip(), city.strip(), neighbourhood.strip(), lat, lon)


# Backward compat aliases
def format_answer_v1(lat: float, lon: float, country: str, city: str) -> str:
    return format_answer_l2(lat, lon, country, city)


def format_answer_v2(lat: float, lon: float, country: str, city: str, neighbourhood: str) -> str:
    return "<answer>{}, {}, {}, {:.4f}, {:.4f}</answer>".format(
        country.strip(), city.strip(), neighbourhood.strip(), lat, lon)


def build_sample(img_path: str, system_prompt: str, user_question: str, answer: str) -> dict:
    """
    Build one verl SFT sample in the correct format:
      - messages[].content is a plain string; user message starts with <image>\\n
      - images is a separate list of {"image": path} dicts
    """
    return {
        "messages": [
            {"role": "user",      "content": "<image>\n{}\n\n{}".format(system_prompt, user_question)},
            {"role": "assistant", "content": answer},
        ],
        "images": [{"image": img_path}],
    }


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------
def get_image_path(img_id: str, images_dir: str) -> Optional[str]:
    """
    Resolve IMG_ID to absolute image path.
    IMG_ID format: XX_YY_nnnnnnnnn.jpg
    Path: <images_dir>/XX/YY/XX_YY_nnnnnnnnn.jpg
    """
    try:
        parts = img_id.split("_")
        dir1 = parts[0]
        dir2 = parts[1]
        path = os.path.join(images_dir, dir1, dir2, img_id)
        return path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------
def process_row(args):
    """
    Process a single CSV row.

    Returns dict with version-keyed messages, or None on failure / missing image.
    """
    idx, row, images_dir, versions = args
    img_id = row["IMG_ID"]

    img_path = get_image_path(img_id, images_dir)
    if img_path is None:
        return None

    try:
        lat = float(row["LAT"])
        lon = float(row["LON"])
    except (ValueError, TypeError):
        return None

    country       = str(row.get("country",       "") or "").strip()
    city          = str(row.get("city",          "") or "").strip()
    region        = str(row.get("region",        "") or "").strip()
    neighbourhood = str(row.get("neighbourhood", "") or "").strip()

    result = {}

    if "l0" in versions:
        cfg = VERSION_CONFIGS["l0"]
        answer = format_answer_l0(lat, lon)
        user_question = random.choice(cfg["question_pool"])
        result["l0"] = build_sample(img_path, cfg["system_prompt"], user_question, answer)

    if "l2" in versions:
        cfg = VERSION_CONFIGS["l2"]
        answer = format_answer_l2(lat, lon, country, city)
        user_question = random.choice(cfg["question_pool"])
        result["l2"] = build_sample(img_path, cfg["system_prompt"], user_question, answer)

    if "l3" in versions:
        cfg = VERSION_CONFIGS["l3"]
        answer = format_answer_l3(lat, lon, country, region, city)
        user_question = random.choice(cfg["question_pool"])
        result["l3"] = build_sample(img_path, cfg["system_prompt"], user_question, answer)

    if "l4" in versions:
        cfg = VERSION_CONFIGS["l4"]
        answer = format_answer_l4(lat, lon, country, region, city, neighbourhood)
        user_question = random.choice(cfg["question_pool"])
        result["l4"] = build_sample(img_path, cfg["system_prompt"], user_question, answer)

    if "v1" in versions:
        cfg = VERSION_CONFIGS["v1"]
        answer = format_answer_v1(lat, lon, country, city)
        user_question = random.choice(cfg["question_pool"])
        result["v1"] = build_sample(img_path, cfg["system_prompt"], user_question, answer)

    if "v2" in versions:
        cfg = VERSION_CONFIGS["v2"]
        answer = format_answer_v2(lat, lon, country, city, neighbourhood)
        user_question = random.choice(cfg["question_pool"])
        result["v2"] = build_sample(img_path, cfg["system_prompt"], user_question, answer)

    return result


# ---------------------------------------------------------------------------
# Batch worker
# ---------------------------------------------------------------------------
def process_batch(batch_args):
    """Process a list of (idx, row) pairs and return a list of result dicts."""
    rows_list, images_dir, versions = batch_args
    results = []
    for idx, row in rows_list:
        try:
            out = process_row((idx, row, images_dir, versions))
            if out is not None:
                results.append(out)
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Records -> PyArrow table (explicit schema, no inference ambiguity)
# ---------------------------------------------------------------------------
def records_to_table(records: list) -> pa.Table:
    """Convert list of {"messages": [...], "images": [...]} dicts to a PyArrow Table using PARQUET_SCHEMA."""
    return pa.Table.from_pylist(records, schema=PARQUET_SCHEMA)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess MP16-Pro -> verl SFT parquet (streaming writes)")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--version", default="all",
        help=(
            "Comma-separated list of versions to generate, or 'all'. "
            "Choices: l0, l2, l3, l4, v1, v2, all, both (both=v1,v2 for backward compat). "
            "Example: --version l0,l2,l3,l4"
        ),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Parse version list ──────────────────────────────────────────────────
    ALL_GRANULARITY = ["l0", "l2", "l3", "l4"]
    VALID_VERSIONS  = {"l0", "l2", "l3", "l4", "v1", "v2"}

    if args.version == "all":
        versions = ALL_GRANULARITY
    elif args.version == "both":
        versions = ["v1", "v2"]
    else:
        versions = [v.strip() for v in args.version.split(",")]
        invalid = set(versions) - VALID_VERSIONS
        if invalid:
            parser.error("Unknown version(s): {}. Valid: {}".format(invalid, VALID_VERSIONS))

    log.info("Generating versions: %s", versions)
    log.info("Images dir: %s", args.images_dir)

    # ── Prepare output dirs and open ParquetWriters ─────────────────────────
    writers = {}   # {version: {"train": ParquetWriter, "test": ParquetWriter}}
    for version in versions:
        version_dir = os.path.join(args.output_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        writers[version] = {
            "train": pq.ParquetWriter(os.path.join(version_dir, "train.parquet"), PARQUET_SCHEMA),
            "test":  pq.ParquetWriter(os.path.join(version_dir, "test.parquet"),  PARQUET_SCHEMA),
        }
        log.info("[%s] Opened parquet writers: %s", version, version_dir)

    # ── Load CSV ────────────────────────────────────────────────────────────
    log.info("Loading CSV: %s", args.csv)
    df = pd.read_csv(args.csv, low_memory=False)
    log.info("Loaded %d rows", len(df))

    # ── 4-level completeness filter (required for l3, l4, and all) ──────────
    needs_4level = any(v in versions for v in ("l3", "l4")) or args.version == "all"
    if needs_4level:
        before = len(df)
        mask = (
            df["country"].notna() & (df["country"] != "") &
            df["region"].notna()  & (df["region"]  != "") &
            df["city"].notna()    & (df["city"]     != "") &
            df["neighbourhood"].notna() & (df["neighbourhood"] != "")
        )
        df = df[mask].copy()
        log.info(
            "4-level completeness filter (country+region+city+neighbourhood): "
            "%d -> %d rows (dropped %d)",
            before, len(df), before - len(df)
        )

    # ── Sample ──────────────────────────────────────────────────────────────
    if args.max_samples and args.max_samples < len(df):
        df = df.sample(n=args.max_samples, random_state=args.seed)
        log.info("Sampled %d rows (seed=%d)", len(df), args.seed)

    before = len(df)
    df = df.dropna(subset=["IMG_ID", "LAT", "LON"])
    log.info("After dropping missing LAT/LON/IMG_ID: %d rows (dropped %d)", len(df), before - len(df))

    rows_list = list(df.iterrows())

    batches = []
    for i in range(0, len(rows_list), args.batch_size):
        batches.append((rows_list[i: i + args.batch_size], args.images_dir, versions))

    log.info("Processing %d rows in %d batches with %d workers (streaming writes)...",
             len(rows_list), len(batches), args.workers)

    rng = random.Random(args.seed)
    stats     = {v: {"train": 0, "test": 0} for v in versions}
    skipped   = 0
    completed = 0
    examples  = {v: [] for v in versions}   # up to 20 train records per version

    log_every = max(1, len(batches) // 20)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_batch, b): i for i, b in enumerate(batches)}

        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                results = future.result()
            except Exception as e:
                log.warning("Batch %d failed: %s", batch_idx, e)
                traceback.print_exc()
                completed += 1
                continue

            skipped += (args.batch_size - len(results))  # rough; last batch may be shorter

            # Split results per version and write immediately
            for version in versions:
                train_recs = []
                test_recs  = []
                for rec in results:
                    if version not in rec:
                        continue
                    entry = {
                        "messages": rec[version]["messages"],
                        "images":   rec[version]["images"],
                    }
                    if rng.random() < args.test_ratio:
                        test_recs.append(entry)
                    else:
                        train_recs.append(entry)
                        if len(examples[version]) < 20:
                            examples[version].append(entry)

                if train_recs:
                    writers[version]["train"].write_table(records_to_table(train_recs))
                    stats[version]["train"] += len(train_recs)

                if test_recs:
                    writers[version]["test"].write_table(records_to_table(test_recs))
                    stats[version]["test"] += len(test_recs)

            completed += 1
            if completed % log_every == 0:
                summary = ", ".join(
                    "[{}] train={} test={}".format(v, stats[v]["train"], stats[v]["test"])
                    for v in versions
                )
                log.info("Progress: %d/%d batches done | %s", completed, len(batches), summary)

    # ── Close writers ────────────────────────────────────────────────────────
    for version in versions:
        writers[version]["train"].close()
        writers[version]["test"].close()
        log.info("[%s] Writers closed. train=%d test=%d",
                 version, stats[version]["train"], stats[version]["test"])

    # ── Save examples ────────────────────────────────────────────────────────
    for version in versions:
        examples_path = os.path.join(args.output_dir, version, "examples.json")
        with open(examples_path, "w", encoding="utf-8") as f:
            json.dump(examples[version], f, indent=2, ensure_ascii=False)
        log.info("[%s] Examples saved: %s", version, examples_path)

    log.info("Done. Output dir: %s", args.output_dir)
    for version in versions:
        log.info("  %s/ train=%d test=%d", version, stats[version]["train"], stats[version]["test"])


if __name__ == "__main__":
    main()
