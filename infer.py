# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import infer
from DB import Database

from color import Color
from daisy import Daisy
from edge  import Edge
from gabor import Gabor
from HOG   import HOG
from vggnet import VGGNetFeat
from resnet import ResNetFeat

import operator
import os
from PIL import Image

depth = 5
d_type = 'd1'
query_idx = 0

def findresult(result):
  fresult = [x['cls'] for x in result]
  dict = {} 
  count, itm = 0, '' 
  for item in reversed(fresult): 
      dict[item] = dict.get(item, 0) + 1
      if dict[item] >= count : 
          count, itm = dict[item], item 
  return(itm) 

def inc(map, key):
  if key not in map.keys():
    map[key] = 1
  else:
    map[key] +=  1

def feed(img):
  db = Database()

  # retrieve by color
  method = Color()

  query = {
    'img': img,
    'cls': 'none'
  }

  query['hist'] = method.make_histogram(query['img'])

  results = {}

  samples = method.make_samples(db)
  # query = samples[query_idx]
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  inc(results, findresult(result))

  # retrieve by daisy
  method = Daisy()
  samples = method.make_samples(db)
  query['hist'] = method.make_histogram(query['img']) # update query with new histogram
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  inc(results, findresult(result))

  # retrieve by edge
  method = Edge()
  samples = method.make_samples(db)
  query['hist'] = method.make_histogram(query['img']) # update query with new histogram
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  inc(results, findresult(result))

  # retrieve by gabor
  # method = Gabor()
  # samples = method.make_samples(db)
  # query['hist'] = method.make_histogram(query['img']) # update query with new histogram
  # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  # print("Gabor: ")
  # print(result)

  # retrieve by HOG
  method = HOG()
  samples = method.make_samples(db)
  query['hist'] = method.make_histogram(query['img']) # update query with new histogram
  _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  inc(results, findresult(result))

  # retrieve by VGG
  # method = VGGNetFeat()
  # samples = method.make_samples(db)
  # query['hist'] = method.make_histogram(query['img']) # update query with new histogram
  # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  # print("VGG: ")
  # print(result)

  # retrieve by resnet
  # method = ResNetFeat()
  # samples = method.make_samples(db)
  # query = samples[query_idx]
  # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
  # print("ResNet: ")
  # print(result)

  finalresult = max(results.items(), key=operator.itemgetter(1))[0]

  print(finalresult)

  string="./database/"+finalresult+"/"
  print(string)
  a=1
  for file in os.listdir(string):
    a+=1
    tempimg=Image.open(string+file)
    tempimg.show()

    print(string+file)
    if (a == 4):
      break


if __name__ == "__main__":
  # feed('test-images/index1.jpeg')
  feed('test-images/index.jpeg')
  # feed('test-images/eli.jpg')