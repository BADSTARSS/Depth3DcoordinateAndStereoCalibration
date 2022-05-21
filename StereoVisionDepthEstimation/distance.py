
import math



def findcenter(p1, p2):

    x1, y1 = p1
    x2, y2 = p2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    return cx,cy

def finddist(pointa,pointa_,cx,cy,dis,d):
    return math.hypot(cx-pointa,cy-pointa_,dis-d)




def find(p1,p2):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = math.hypot(x2 - x1, y2 - y1)
    info = (x1, y1, x2, y2, cx, cy)

    return length