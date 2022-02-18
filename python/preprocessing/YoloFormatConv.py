import os
import cv2
Annotations = os.listdir(r"C:\Users\dzhan\Desktop\Folder\Annotations")
for i in Annotations:
    img = cv2.imread(r'C:\Users\dzhan\Desktop\Folder\Images\\'+i[:10])
    height = img.shape[0]
    width = img.shape[1]
    path = (r"C:\Users\dzhan\Desktop\Folder\Annotations\\" + i)
    def read_text_file(file_path):
        with open(file_path, 'r') as f:
            return f.read()
    txt = read_text_file(path)
    a = txt[0:len(txt)].find('\n')
    b = txt[a+1:]
    lst = []
    for i in b.split('\n')[:-1]:
        c = i[2:].split(' ')
        x1 = (int(c[0]) + int(c[2]))/(2 * width)
        y1 = (int(c[1]) + int(c[3]))/(2 * height)
        x2 = (int(c[2]) - int(c[0]))/(width)
        y2 = (int(c[3]) - int(c[1]))/(height)
        size = [x1,y1,x2,y2]
        one = ' '.join([str(item) for item in size])
        one = '0 ' + one
        lst.append(one)
    new_name = path[43:49]+path[53:]
    f = open(r"C:\Users\dzhan\Desktop\Folder\New_Annotations\\" + new_name, "x")
    with open(r"C:\Users\dzhan\Desktop\Folder\New_Annotations\\" + new_name, "w") as fobj:
        for x in lst:
            fobj.write(x + "\n")
    f.close()
    fd = open(r"C:\Users\dzhan\Desktop\Folder\New_Annotations\\" + new_name,"r")
    d = fd.read()
    fd.close()
    m = d.split("\n")
    s = "\n".join(m[:-1])
    fd = open(r"C:\Users\dzhan\Desktop\Folder\New_Annotations\\" + new_name,"w+")
    for i in range(len(s)):
        fd.write(s[i])
    fd.close()