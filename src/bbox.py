from PIL import Image
import pylab
import sys
sys.path.append(".") 
import open_dl_utils as d2l

d2l.set_figsize()
img = Image.open('/home/linyang/pictures/person.jpg')
d2l.plt.imshow(img)
pylab.show()

dog_bbox, cat_bbox = [60, 45, 178, 216], [200, 112, 255, 293]
fig = d2l.plt.imshow(img)
fig.axes.add_patch(d2l.bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(d2l.bbox_to_rect(cat_bbox, 'red'))
d2l.plt.show()