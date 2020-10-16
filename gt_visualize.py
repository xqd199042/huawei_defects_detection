from xml.dom.minidom import parse, Element
import os
import cv2


def polygon_to_line(list_axis):
    list_x, list_y = [], []
    for i, axis in enumerate(list_axis):
        if i%2==0: list_x.append(axis)
        else: list_y.append(axis)
    min_x, min_y = min(list_x), min(list_y)
    max_x, max_y = max(list_x), max(list_y)
    return (min_x, min_y), (max_x, max_y)

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
            dic['axis'] = ((list_axis[0], list_axis[3]), (list_axis[2], list_axis[1]))
        else:
            polygon = item.getElementsByTagName('polygon')[0]
            list_axis = [int(node.childNodes[0].data) for node in polygon.childNodes if type(node)==Element]
            dic['axis'] = polygon_to_line(list_axis)
        result.append(dic)
    return result

def visualize(img, infos):
    for info in infos:
        a1, a2 = info['axis']
        cv2.rectangle(img, a1, a2, (0,0,255), 2)
        cv2.putText(img, info['name'], (a1[0], a1[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.imwrite('test.jpg', img)

def defect_info_out(xml_folder_path):
    result = {}
    xml_files = os.listdir(xml_folder_path)
    for xml_file in xml_files:
        info_path = os.path.join(xml_folder_path, xml_file)
        infos = xml_info_out(info_path)
        img_path = info_path.replace('side/outputs', 'side').replace('.xml', '.jpg')
        img = cv2.imread(img_path)
        for info in infos:
            name, axis = info['name'], info['axis']
            min_x, max_x = min([axis[0][0], axis[1][0]]), max([axis[0][0], axis[1][0]])
            min_y, max_y = min([axis[0][1], axis[1][1]]), max([axis[0][1], axis[1][1]])
            width, height = max_x-min_x, max_y-min_y
            img_new = img[min_y:max_y+1, min_x:max_x+1]
            if name not in result:
                result[name] = {'count':1, 'width':[width], 'height':[height]}
            else:
                result[name]['count'] += 1
                result[name]['width'].append(width)
                result[name]['height'].append(height)
            cv2.imwrite('out/'+name+str(result[name]['count'])+'.jpg', img_new)
            print(name+str(result[name]['count'])+'.jpg'+' finished!')
    return result


# /home/qiangde/Data/huawei/black/side/
# /home/qiangde/Data/huawei/black/side/outputs/
if __name__ == '__main__':
    img = cv2.imread('/home/qiangde/Data/huawei/black/side/0897-0003-14.jpg')
    infos = xml_info_out('/home/qiangde/Data/huawei/black/side/outputs/0897-0003-14.xml')
    visualize(img, infos)
    result = defect_info_out('/home/qiangde/Data/huawei/black/side/outputs/')
    for key in result:
        print(key+': ',result[key])






























