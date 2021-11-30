from PIL import Image
from glob import glob
import ipdb


files = glob('./data/*/*/*_image.jpg')
files.sort()
#print(files)

for f in files:
    image = Image.open(f)
    image = image.resize((448, 448))
    image.save(f)

    #ipdb.set_trace()
    #image = image.resize(448, 448)
