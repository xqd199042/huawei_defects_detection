import random
from xml.dom.minidom import parse, Element
import os
import cv2


# convert segmentation axis to detection axis
def polygon_to_line(list_axis):
    list_x, list_y = [], []
    for i, axis in enumerate(list_axis):
        if i%2==0: list_x.append(axis)
        else: list_y.append(axis)
    min_x, min_y = min(list_x), min(list_y)
    max_x, max_y = max(list_x), max(list_y)
    return (min_x, min_y), (max_x, max_y)

# given a xml path, return a list [...] containing {defect_name defect_axis}
def xml_info_out(anno_path):
    result = []
    root = parse(anno_path).documentElement
    items = root.getElementsByTagName('item')
    for item in items:
        dic = {}
        name = item.getElementsByTagName('name')[0].childNodes[0].data
        dic['name'] = name
        lines = item.getElementsByTagName('line')
        if len(lines)>0:
            line = lines[0]
            list_axis = [int(node.childNodes[0].data) for node in line.childNodes if type(node)==Element]
            dic['axis'] = polygon_to_line(list_axis)
        else:
            polygon = item.getElementsByTagName('polygon')[0]
            list_axis = [int(node.childNodes[0].data) for node in polygon.childNodes if type(node)==Element]
            dic['axis'] = polygon_to_line(list_axis)
        result.append(dic)
    return result

# given a xml infos, visualize the defects on img
def visualize(img, xml_infos):
    for info in xml_infos:
        a1, a2 = info['axis']
        cv2.rectangle(img, a1, a2, (0,0,255), 2)
        cv2.putText(img, info['name'], (a1[0], a1[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.imwrite('test.jpg', img)

# return all the information and cropped defects
def defect_infos_out(xml_folder_path, crop=False, crop_size=128):
    result = {}
    xml_files = os.listdir(xml_folder_path)
    for xml_file in xml_files:
        info_path = os.path.join(xml_folder_path, xml_file)
        infos = xml_info_out(info_path)
        for info in infos:
            name, axis = info['name'], info['axis']
            min_x, min_y = axis[0]
            max_x, max_y = axis[1]
            center_x, center_y = (min_x+max_x)//2, (min_y+max_y)//2
            width, height = max_x-min_x+1, max_y-min_y+1
            if name not in result:
                result[name] = {'count':1, 'width':[width], 'height':[height]}
            else:
                result[name]['count'] += 1
                result[name]['width'].append(width)
                result[name]['height'].append(height)
            if crop:
                img_path = info_path.replace('side/outputs', 'side').replace('.xml', '.jpg')
                img = cv2.imread(img_path)
                crop_min_y, crop_max_y, crop_min_x, crop_max_x = center_y-crop_size//2, center_y+crop_size//2, center_x-crop_size//2, center_x+crop_size//2
                if center_y-crop_size//2 <= 0: crop_min_y, crop_max_y = 0, crop_size
                if center_y+crop_size//2 >= 14600: crop_min_y, crop_max_y = 14600-crop_size, 14600
                if center_x-crop_size//2 <= 0: crop_min_x, crop_max_x = 0, crop_size
                if center_x+crop_size//2 >= 800: crop_min_x, crop_max_x = 800-crop_size, 800
                img_new = img[crop_min_y:crop_max_y, crop_min_x:crop_max_x]
                cv2.imwrite('out/'+name+str(result[name]['count'])+'.jpg', img_new)
                print(name+str(result[name]['count'])+'.jpg'+' finished!')
    return result

# this class is used to calculate iou and anchor clustering
class Box:
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    def get_area(self):
        return abs(self.max_x-self.min_x)*abs(self.max_y-self.min_y)

# this class is used to calculate iou and anchor clustering
class Anchor(Box):
    def __init__(self, init_w, init_h):
        super(Anchor, self).__init__(800-init_w/2, 10000-init_h/2, 800+init_w/2, 10000+init_h/2)
        self.boxes = []
        self.num = 0
        self.avg_iou = 0
        self.width = init_w
        self.height = init_h

    def update_wh(self):
        w, h = 0, 0
        ious = 0
        self.num = len(self.boxes)
        for box in self.boxes:
            w += box.max_x-box.min_x
            h += box.max_y-box.min_y
            ious += compute_iou(box, self)
        self.width = w/self.num
        self.height = h/self.num
        self.min_x = 800 - self.width/2
        self.min_y = 10000 - self.height/2
        self.max_x = 800 + self.width/2
        self.max_y = 10000 + self.height/2
        self.avg_iou = ious/self.num
        self.boxes = []

# return box list, center is (800, 10000)
def to_boxes(list_w, list_h):
    result = []
    for i, w in enumerate(list_w):
        h = list_h[i]
        min_x, min_y, max_x, max_y = 800-w/2, 10000-h/2, 800+w/2, 10000+h/2
        result.append(Box(min_x,min_y,max_x,max_y))
    return result

def compute_iou(box1: Box, box2: Box):
    area1, area2 = box1.get_area(), box2.get_area()
    inter_box = Box(max([box1.min_x, box2.min_x]), max([box1.min_y, box2.min_y]), min([box1.max_x, box2.max_x]), min([box1.max_y, box2.max_y]))
    inter_area = inter_box.get_area()
    return inter_area/(area1+area2-inter_area)

def kmeans_update(boxes, anchors):
    for box in boxes:
        temp = []
        for anchor in anchors:
            iou = compute_iou(box, anchor)
            temp.append(iou)
        index = temp.index(max(temp))
        anchors[index].boxes.append(box)
    for i, anchor in enumerate(anchors):
        print('the %dth anchor:' % (i+1))
        anchor.update_wh()
        print('num_boxes: %d, avg_iou: %f, width: %f, height: %f'%(anchor.num, anchor.avg_iou, anchor.width, anchor.height))

# /home/qiangde/Data/huawei/black/side/
# /home/qiangde/Data/huawei/black/side/outputs/
if __name__ == '__main__':
    # img = cv2.imread('/home/qiangde/Data/huawei/black/side/0897-0003-14.jpg')
    # infos = xml_info_out('/home/qiangde/Data/huawei/black/side/outputs/0897-0003-14.xml')
    # visualize(img, infos)
    result = defect_infos_out('/home/qiangde/Data/huawei/black/side/outputs/')
    boxes = []
    anchors = []
    for key in result:
        list_w, list_h = result[key]['width'], result[key]['height']
        boxes.extend(to_boxes(list_w, list_h))
    selected_boxes = random.sample(boxes, 9)
    for b in selected_boxes:
        anchors.append(Anchor(b.max_x-b.min_x, b.max_y-b.min_y))
    for i in range(100):
        print('Iteration %d' % (i+1))
        kmeans_update(boxes, anchors)

    ious = 0
    for anchor in anchors:
        ious += anchor.avg_iou*anchor.num
    print(ious/len(boxes))






























