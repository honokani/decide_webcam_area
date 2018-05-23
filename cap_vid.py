import cv2
import copy
import numpy as np

CUTTING     = 0
M_PUSHING   = 0
S_POS_X     = 0
S_POS_Y     = 0
E_POS_X     = 0
E_POS_Y     = 0

SQUARE_MODE = 0
SQUARE_STRICT_MODE = 0

CONST_MODE  = 0
CONST_W     = 0

def keep_status_s(x,y):
    global CUTTING
    global M_PUSHING
    global S_POS_X
    global S_POS_Y
    S_POS_X = x
    S_POS_Y = y
    keep_status_e(x,y)
    M_PUSHING = 1
    CUTTING   = 1
    print("s: x={}, y={}, f={}".format(S_POS_X,S_POS_Y,M_PUSHING))

def keep_status_e(x,y):
    global E_POS_X
    global E_POS_Y
    E_POS_X = x
    E_POS_Y = y

def mouse_event(e, x, y, f, p):
    global M_PUSHING
    if(M_PUSHING==0 and e==cv2.EVENT_LBUTTONDOWN):
        print("l_down!")
        keep_status_s(x,y)
    elif(M_PUSHING==1):
        if(e==cv2.EVENT_LBUTTONUP):
            print("l_up!")
            M_PUSHING = 0
        else:
            keep_status_e(x,y)

def culcSquareArea(fx, fy):
    dx = E_POS_X - S_POS_X
    sx = 1 if (0<dx) else -1
    dx = abs(dx)
    dy = E_POS_Y - S_POS_Y
    sy = 1 if (0<dy) else -1
    dy = abs(dy)
    px = 0
    py = 0
    lx = S_POS_X if (sx < 0) else fx-S_POS_X
    ly = S_POS_Y if (sy < 0) else fy-S_POS_Y

    if (dy < dx):
        if(ly < dx or fx < E_POS_X):
            dx = ly if (ly < lx) else lx
            px = S_POS_X + sx * dx
        elif(E_POS_X < 0):
            dx = S_POS_X
            px = 0
        else:
            px = E_POS_X
        py = S_POS_Y + sy * dx
    else:
        if(lx < dy or fy < E_POS_Y):
            dy = lx if (lx < ly) else ly
            py = S_POS_Y + sy * dy
        elif(E_POS_Y < 0):
            dy = S_POS_Y
            py = 0
        else:
            py = E_POS_Y
        px = S_POS_X + sx * dy

    return px, py

def makeTags(x,y):
    h = 16
    w = (1+len(list(map(int,str(x))))+len(list(map(int,str(y)))))*10+2
    tagImg = makeWindow(h,w,(255,0,0))
    tagImg = cv2.putText(tagImg, "{}x{}".format(x,y), (0,15), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, 4)
    return tagImg

def clip_image(b,f,x,y):
    h = f.shape[0]
    w = f.shape[1]
    b[y:y+h,x:x+w] = f
    return b

def makeWindow(sx,sy,col):
    b,g,r = col
    tagImg = np.zeros((sx,sy,3), np.uint8)
    for i in range(sx):
        for j in range(sy):
            tagImg[i,j] = [b,g,r]
    return tagImg

def setConstParam(x,y,sn):
    msg = [ "(cancel:q, erase:BS, decise:Enter)"
          , "input size (0 ~ {}): {}".format(x if x<y else y, sn)
          ]
    l = 0
    for  m in msg:
        c = len(m)
        l = c if (l < c) else l
    h = 16*len(msg)
    w = 10*l+2
    iImg = makeWindow(h,w,(255,255,255))
    for i, m in enumerate(msg):
        iImg = cv2.putText(iImg, m, (0,15+i*16), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, 4)
    return iImg

def setHelpParam():
    msg = [ "h: help"
          , "q: quit"
          , "s: square area mode (and switch normal mode)"
          , "r: release area"
          , "c: const area mode // TODO"
          ]
    l = 0
    for  m in msg:
        c = len(m)
        l = c if (l < c) else l
    h = 16*len(msg)
    w = 10*l+2
    iImg = makeWindow(h,w,(255,255,255))
    for i, m in enumerate(msg):
        iImg = cv2.putText(iImg, m, (0,15+i*16), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1, 4)
    return iImg


def main():
    global CUTTING
    global SQUARE_MODE
    global CONST_MODE
    global CONST_W
    # 動画の読み込み
    cap = cv2.VideoCapture(0)

    # 動画終了まで繰り返し
    frame = 0
    f_sz_x = 0
    f_sz_y = 0
    while(cap.isOpened()):
        # フレームを取得
        if(not M_PUSHING==1):
            ret, frame = cap.read()
            f_sz_y = frame.shape[0]
            f_sz_x = frame.shape[1]
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Frame", mouse_event)

        if(CUTTING==0):
            fixed_frame = frame
        elif(CONST_MODE == 1):
            edit_frame = copy.copy(frame)
        elif(CUTTING == 1):
            edit_frame = copy.copy(frame)
            if(M_PUSHING==1):
                if(SQUARE_MODE == 1):
                    px, py = culcSquareArea(f_sz_x, f_sz_y)
                else:
                    px = 0 if (E_POS_X < 0) else (f_sz_x if (f_sz_x < E_POS_X) else E_POS_X )
                    py = 0 if (E_POS_Y < 0) else (f_sz_y if (f_sz_y < E_POS_Y) else E_POS_Y )
                cv2.line(edit_frame, (S_POS_X,S_POS_Y), (S_POS_X,py), (0,0,255), 2)
                cv2.line(edit_frame, (S_POS_X,S_POS_Y), (px,S_POS_Y), (0,0,255), 2)
                cv2.line(edit_frame, (px,S_POS_Y), (px,py), (0,0,255), 2)
                cv2.line(edit_frame, (S_POS_X,py), (px,py), (0,0,255), 2)
                tx = abs(px-S_POS_X)
                tagImg = makeTags(tx,abs(py-S_POS_Y))
                edit_frame = clip_image(edit_frame, tagImg, 0, 0)
                fixed_frame = edit_frame


        # qキーが押されたら途中終了
        k = cv2.waitKey(1)
        if(k & 0xFF == ord('q')):
            print("q!")
            break
        elif(k & 0xFF == ord('r')):
            print("r!")
            CUTTING = 0
        elif(k & 0xFF == ord('s')):
            print("s!")
            SQUARE_MODE = 1 ^ SQUARE_MODE
        elif(k & 0xFF == ord('h')):
            print("h!")
            helpWindowImg = setHelpParam()
            while(True):
                k = cv2.waitKey(1)
                fixed_frame = clip_image(fixed_frame, helpWindowImg, 0, 0)
                cv2.imshow("Frame", fixed_frame)
                if(k & 0xFF == ord('q')):
                    fixed_frame = frame
                    break
        elif(k & 0xFF == ord('c')):
            print("c!")
            CONST_MODE = 1 ^ CONST_MODE
            if(CONST_MODE==1):
                size_num = ""
                isDecided = 0
                while(True):
                    k = cv2.waitKey(1)
                    if(k & 0xFF == ord('q')):
                        CONST_MODE = 0
                        break
                    elif(k & 0xFF == ord('1')):
                        size_num = size_num + "1"
                    elif(k & 0xFF == ord('2')):
                        size_num = size_num + "2"
                    elif(k & 0xFF == ord('3')):
                        size_num = size_num + "3"
                    elif(k & 0xFF == ord('4')):
                        size_num = size_num + "4"
                    elif(k & 0xFF == ord('5')):
                        size_num = size_num + "5"
                    elif(k & 0xFF == ord('6')):
                        size_num = size_num + "6"
                    elif(k & 0xFF == ord('7')):
                        size_num = size_num + "7"
                    elif(k & 0xFF == ord('8')):
                        size_num = size_num + "8"
                    elif(k & 0xFF == ord('9')):
                        size_num = size_num + "9"
                    elif(k & 0xFF == 0x08):
                        if(not size_num == ""):
                            size_num = size_num[:-1]
                    elif(k & 0xFF == 0x0d):
                        isDecided = 1

                    if(isDecided == 0):
                        inWindowImg = setConstParam(f_sz_x,f_sz_y,size_num)
                        fixed_frame = clip_image(fixed_frame, inWindowImg, 0, 0)
                        cv2.imshow("Frame", fixed_frame)
                    else:
                        if(0<len(size_num)):
                            CONST_W = int(size_num)
                        else:
                            CONST_MODE = 0
                        break

        elif(k & 0xFF == ord('w')):
            print("w!")


        # フレームを表示
        cv2.imshow("Frame", fixed_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("start")
    main()
    print("end")

