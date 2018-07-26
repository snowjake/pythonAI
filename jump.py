import pymouse, time, pyHook, pythoncom, math, random

m = pymouse.PyMouse()
global start_pos, end_pos
start_pos = None
end_pos = None

def onKeyboardEvent(event):
    global start_pos, end_pos
    if event.Key == 'Q':
        start_pos = m.position()
        print(start_pos)
    if event.Key == 'W':
        end_pos = m.position()
        print(end_pos)
    if event.Key == 'Space':
        if start_pos and end_pos:
            dis = int(math.sqrt(math.pow(start_pos[0]-end_pos[0],2)+math.pow(start_pos[1]-end_pos[1],2)))
            times = round((dis / 0.3) / 1000 , 3) 
            if dis < 500:
                print ('>> ', dis, times)
                x = random.randint(50, 400)
                y = random.randint(400, 700)
                m.press(x,y)
                time.sleep(times)
                m.release(x,y)
                start_pos = None
                end_pos = None
    return True

def main():
    hm = pyHook.HookManager()
    hm.KeyDown = onKeyboardEvent
    try:
        hm.HookKeyboard()
    except Exception as e:
        pass
    pythoncom.PumpMessages()
if __name__ == "__main__":
    main()