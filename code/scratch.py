import glob
import numpy as np
from pprint import pprint as pp


g = glob.glob("data/*/*/audio/*.wav")

wavpath = g[:10]

pp(wavpath)

res = [("/".join(wp.split("/")[:-2]), "/".join(wp.split("/")[-2:])) for wp in wavpath]

pp(res)