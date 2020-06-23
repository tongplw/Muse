import time
from muse import MuseMonitor


SERVER = "192.168.1.122"
PORT = 5000

if __name__ == "__main__":
    muse = MuseMonitor(SERVER, PORT)
    
    while True:
        print(muse.raw)
        print(muse.waves)
        print(muse.attention)
        time.sleep(1)