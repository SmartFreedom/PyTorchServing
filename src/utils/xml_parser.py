import os
import easydict
import pandas as pd
import xml.etree.ElementTree as et
import urllib.request
import json

from ..configs import config


def get_sort_order(fileid):
    url = str(os.path.join(config.API.CASES, '{}?api_key={}'))
    jd = json.JSONDecoder()

    req = urllib.request.Request(url.format(fileid, config.API.KEY))
    with urllib.request.urlopen(req) as response:
        body = response.read()
        body = body.decode("utf-8")
        body = jd.decode(body)

    return [ 
        { 'ID': el['id'], 'slice': el['sort_order'] } 
        for el in body['files'] 
    ]


def parse_occlusion(obj):
    occluded = obj.find('{http://www.w3.org/1999/xhtml}occluded')
    return occluded.text


def parse_name(obj):
    name = obj.find('{http://www.w3.org/1999/xhtml}name')
    return name.text


def parse_polygon(obj):
    polygon = obj.find('{http://www.w3.org/1999/xhtml}polygon')
    points = polygon.findall('{http://www.w3.org/1999/xhtml}pt')

    coords = list()
    for point in points:
        x = point.find('{http://www.w3.org/1999/xhtml}x')
        y = point.find('{http://www.w3.org/1999/xhtml}y')
        x, y = x.text, y.text
        coords.extend((int(x), int(y)))
    return coords#' '.join(coords)


def compose_dataframe(data, row):
    data = pd.DataFrame(data)
    data['id'] = row.ID
    return data


def parse_xml(row):
    xml_tree = et.fromstring(row.XML)
    objects = xml_tree.findall('{http://www.w3.org/1999/xhtml}object')

    data = list()

    for obj in objects:
        occlusion = parse_occlusion(obj)
        coords = parse_polygon(obj)
        name = parse_name(obj)

        data.append({
            'occluded': occlusion, 
            'coords': coords, 
            'name': name,
        })
    return compose_dataframe(data, row)


def parse_annotations(annotations):
    data = list()

    for i, row in annotations.iterrows():
        objects = et.fromstring(row.XML)
        data.extend([
            easydict.EasyDict({ 
                'ID': ch.find('fileid').text,
                'id': obj.find('{http://www.w3.org/1999/xhtml}id').text,
                'name': obj.find('{http://www.w3.org/1999/xhtml}name').text,
                'case': row['Кейс'],
                'elem': ch,
                'object': obj,
                'coords': parse_polygon(obj),
                'filename': ch.find('filename').text,
            })
            for i, ch in enumerate(sorted(list(objects), key=lambda x: int(x.find('fileid').text))) 
            for obj in ch.findall('{http://www.w3.org/1999/xhtml}object')
            if ch.findall('private') 
        ])

    return pd.DataFrame([ d for d in data if d])
