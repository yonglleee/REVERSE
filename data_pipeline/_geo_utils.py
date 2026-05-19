"""GPS 提取、坐标转换、反向地理编码、搜索 API、GCS 上传。

从 search_agent/tools.py 内化，供 kimi_tool_call 包内使用。
"""

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from serpapi import BaiduSearch, GoogleSearch  # pip install google-search-results

# ── WGS-84 椭球参数 ──────────────────────────────────────────────────────────

_a = 6378245.0
_ee = 0.00669342162296594

_OXYLABS_PROXY = None
_OXYLABS_PROXIES = {'http': _OXYLABS_PROXY, 'https': _OXYLABS_PROXY}

_OXYLABS_PROXIES = None


def dms_to_decimal(dms, ref):
    """将度分秒 (DMS) 转换为十进制度数"""
    degrees, minutes, seconds = dms
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in ('S', 'W'):
        decimal = -decimal
    return decimal


def _out_of_china(lat, lon):
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lat, lon):
    """WGS-84 转 GCJ-02（火星坐标系）"""
    if _out_of_china(lat, lon):
        return lat, lon
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    rad_lat = lat / 180.0 * math.pi
    magic = math.sin(rad_lat)
    magic = 1 - _ee * magic * magic
    sqrt_magic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_a * (1 - _ee)) / (magic * sqrt_magic) * math.pi)
    dlon = (dlon * 180.0) / (_a / sqrt_magic * math.cos(rad_lat) * math.pi)
    return round(lat + dlat, 6), round(lon + dlon, 6)


# ── GPS 提取 ─────────────────────────────────────────────────────────────────

def extract_gps(img_path):
    """从图片 EXIF 中提取 GPS 经纬度（GCJ-02 坐标系）"""
    img = Image.open(img_path)
    exif_data = img._getexif()
    if not exif_data:
        print("图片不包含 EXIF 数据")
        return None

    gps_info = {}
    for tag_id, value in exif_data.items():
        tag_name = TAGS.get(tag_id, tag_id)
        if tag_name == 'GPSInfo':
            for gps_tag_id, gps_value in value.items():
                gps_tag_name = GPSTAGS.get(gps_tag_id, gps_tag_id)
                gps_info[gps_tag_name] = gps_value

    if not gps_info:
        print("图片不包含 GPS 信息")
        return None

    lat = dms_to_decimal(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
    lon = dms_to_decimal(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
    gcj_lat, gcj_lon = wgs84_to_gcj02(lat, lon)

    return gcj_lat, gcj_lon


# ── 反向地理编码 ──────────────────────────────────────────────────────────────

def restapi_reverse_geocode(longitude: float, latitude: float):
    """调用高德地图反向地理编码 API，返回可读地址字符串或 None。"""
    amap_key = os.environ.get('AMAP_API_KEY')
    if not amap_key:
        raise ValueError('AMAP_API_KEY 环境变量未设置')
    url = (
        f'https://restapi.amap.com/v3/geocode/regeo?output=xml'
        f'&location={longitude},{latitude}'
        f'&key={amap_key}&radius=50&extensions=base'
    )

    response = requests.get(url)
    if response.status_code != 200:
        return None

    content = response.content.decode('utf-8')

    def extract_tag(tag, text=content):
        m = re.search(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
        return m.group(1).strip() if m else None

    status = extract_tag('status')
    if status != '1':
        return None

    info = extract_tag('info')
    if info != 'OK':
        return None

    regeocode = extract_tag('regeocode')
    formatted_address = extract_tag('formatted_address', regeocode)
    return formatted_address


# ── GCS 上传 ──────────────────────────────────────────────────────────────────

_GCS_CREDENTIALS_PATH = '/mnt/sh/mmvision/home/jonahli/projects/tusou/data_pipeline/image_search/wired-epsilon-418710-c4a22b0d482e.json'
_GCS_BUCKET_NAME = 'geo-search-agent'

_IMAGE_CONTENT_TYPES = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.gif': 'image/gif',
    '.webp': 'image/webp',
    '.bmp': 'image/bmp',
}


def upload_image_to_gcs(local_path: str, destination_blob_name=None) -> str:
    """上传本地图片到 Google Cloud Storage，返回 public URL。"""
    import uuid
    from google.cloud import storage as gcs_storage

    os.environ.setdefault('GOOGLE_APPLICATION_CREDENTIALS', _GCS_CREDENTIALS_PATH)

    if not os.path.isfile(local_path):
        raise FileNotFoundError(f'文件不存在: {local_path}')

    if destination_blob_name is None:
        ext = os.path.splitext(local_path)[1].lower()
        destination_blob_name = f'geo_agent/{uuid.uuid4().hex}{ext}'

    client = gcs_storage.Client()
    bucket = client.bucket(_GCS_BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)

    ext = os.path.splitext(local_path)[1].lower()
    content_type = _IMAGE_CONTENT_TYPES.get(ext, 'application/octet-stream')
    blob.upload_from_filename(local_path, content_type=content_type)

    public_url = f'https://storage.googleapis.com/{_GCS_BUCKET_NAME}/{destination_blob_name}'
    return public_url


# ── SerpAPI 搜索 ─────────────────────────────────────────────────────────────

_SEARCH_API_KEY = os.environ.get('SERPER_API_KEY')


def serper_search_img(image_url: str) -> list:
    """根据图片 URL 调用 Google Lens 反向图搜，返回 visual_matches 列表。"""
    if not _SEARCH_API_KEY:
        raise ValueError('SERPER_API_KEY 环境变量未设置')
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": _SEARCH_API_KEY,
    }

    search = GoogleSearch(params)
    # print('search', search)
    results = search.get_dict()
    # print('results', results)
    visual_matches = results["visual_matches"][:5]
    return visual_matches


def serper_search_text(keywords) -> list:
    """并行调用 Google + Baidu 文本搜索，合并返回 Top 结果。"""

    def _google_search():
        params = {
            "api_key": _SEARCH_API_KEY,
            "engine": "google_light",
            "google_domain": "google.com",
            "q": keywords,
            "hl": "zh-cn",
            "gl": "cn",
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results.get('organic_results', [])[:3]

    def _baidu_search():
        params = {
            "engine": "baidu",
            "q": keywords,
            "api_key": _SEARCH_API_KEY,
        }
        search = BaiduSearch(params)
        results = search.get_dict()
        return results.get("organic_results", [])[:3]

    with ThreadPoolExecutor(max_workers=2) as executor:
        google_future = executor.submit(_google_search)
        baidu_future = executor.submit(_baidu_search)
        google_results = google_future.result()
        baidu_results = baidu_future.result()

    return google_results + baidu_results



#——————————————————————oxylabs API -------

def oxylabs_reverse_image_search_img(image_url: str) -> list:
    """根据图片 URL 调用 Oxylabs API，返回 visual_matches 列表。"""
    oxylabs_user = os.environ.get('OXYLABS_USER')
    oxylabs_pass = os.environ.get('OXYLABS_PASSWORD')
    if not (oxylabs_user and oxylabs_pass):
        raise ValueError('OXYLABS_USER 或 OXYLABS_PASSWORD 环境变量未设置')

    payload = {
    'source': 'google_lens',
    'query': image_url,
    'parse': True,
    'locale': 'zh-cn-us',

}
    response = requests.request(
    'POST',
    'https://realtime.oxylabs.io/v1/queries',
    auth=(oxylabs_user, oxylabs_pass),
    json=payload,
    proxies=_OXYLABS_PROXIES,
)

    result_raw = response.json()
    results_top10 = result_raw['results'][0]['content']['results']['organic'][:10]

    result_formated = [{
        "position": x['pos'],
        "title": x['title'],
        "link": x['url'],
        "source": x['domain'],
        "thumbnail": x['url_thumbnail'],
        "image": x['url_thumbnail'],
    } for x in results_top10]
    
    return result_formated


def oxylabs_search_text_organic(keywords) -> list:

    oxylabs_user = os.environ.get('OXYLABS_USER')
    oxylabs_pass = os.environ.get('OXYLABS_PASSWORD')
    if not (oxylabs_user and oxylabs_pass):
        raise ValueError('OXYLABS_USER 或 OXYLABS_PASSWORD 环境变量未设置')

    # general google text search
    payload = {
    'source': 'google_search',
    'query': keywords,
    'locale': 'zh-cn-us',
    'geo_location': 'Singapore',
    'parse': True
}
    response = requests.request(
    'POST',
    'https://realtime.oxylabs.io/v1/queries',
    auth=(oxylabs_user, oxylabs_pass),
    json=payload,
    proxies=_OXYLABS_PROXIES,
)

    results_raw = response.json()

    organic_results_formated = []
    top_sights_formated = []
    twitter_formated = []
    top_stories_top3_formated = []

    if 'organic' in results_raw['results'][0]['content']['results']:
        organic_results_top10 = results_raw['results'][0]['content']['results']['organic'][:5]
        # print(organic_results_top10)
        organic_results_formated = [{
            "position": x['pos'],
            "title": x['title'],
            "link": x['url'],
            'images':x.get('images',[]),
            "snippet": x['desc'],
        } for x in organic_results_top10]
    

    if 'top_sights' in results_raw['results'][0]['content']['results']:
        top_sights_top10 = results_raw['results'][0]['content']['results']['top_sights'][:2]
        top_sights_formated = [{
            "position": x['pos'],
            "title": x['title']
        } for x in top_sights_top10]


    if 'twitter' in results_raw['results'][0]['content']['results']:
        twitter_top3 = results_raw['results'][0]['content']['results']['twitter']['items'][:2]
        twitter_formated = [{
            "link": x['url'],
            "title": x['content'],
        } for x in twitter_top3]

    
    if 'top_stories' in results_raw['results'][0]['content']['results']:
        top_stories_top3 = results_raw['results'][0]['content']['results']['top_stories']['items'][:2]
        top_stories_top3_formated = [
            {'title':x['title']} for x in top_stories_top3
        ]

    return organic_results_formated + top_sights_formated + twitter_formated + top_stories_top3_formated


def oxylabs_search_text_img(keyword):
    """根据关键词调用 google 图搜 API，返回 google 检索图片 Top 15 的图片+title列表。"""

    oxylabs_user = os.environ.get('OXYLABS_USER')
    oxylabs_pass = os.environ.get('OXYLABS_PASSWORD')
    if not (oxylabs_user and oxylabs_pass):
        raise ValueError('OXYLABS_USER 或 OXYLABS_PASSWORD 环境变量未设置')

    payload = {
        'source': 'google_search',
        'query': keyword,
        'parse': True,
        'context': [
            {'key': 'udm', 'value': 2},
        ],
        'locale': 'zh-cn-us',
        'geo_location': 'Singapore',
    }

    response = requests.post(
        'https://realtime.oxylabs.io/v1/queries',
        auth=(oxylabs_user, oxylabs_pass),
        json=payload,
        proxies=_OXYLABS_PROXIES,
    )

    response_raw = response.json()
    organic = (
        response_raw.get('results', [{}])[0]
        .get('content', {})
        .get('results', {})
        .get('organic', [])
    )
    results_top15 = organic[:15]

    text_img_top15_formated = [{
        "title": x.get('title', ''),
        "link": x.get('url', x.get('link', '')),
        "image": x.get('image', x.get('url_image', x.get('url_thumbnail', ''))),
    } for x in results_top15]

    # print('text_img_top15_formated',text_img_top15_formated[0])
    return text_img_top15_formated

def oxylabs_fetch_html(url):
    oxylabs_user = os.environ.get('OXYLABS_USER')
    oxylabs_pass = os.environ.get('OXYLABS_PASSWORD')
    if not (oxylabs_user and oxylabs_pass):
        raise ValueError('OXYLABS_USER 或 OXYLABS_PASSWORD 环境变量未设置')

    payload = {
    'source': 'universal',
    'url': url,
    'render': 'html', # If page type requires
}

    # Get response.
    response = requests.request(
        'POST',
        'https://realtime.oxylabs.io/v1/queries',
        auth=(oxylabs_user, oxylabs_pass),
        json=payload,
        proxies=_OXYLABS_PROXIES,
    )

    result_raw = response.json()
    result_html = result_raw['results'][0]['content']
    return result_html


if __name__ == '__main__':
    text_img_top15_formated = oxylabs_search_text_img('灵隐寺 华严殿 毗卢遮那佛 杭州')
    print('text_img_top15_formated',text_img_top15_formated)