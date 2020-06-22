from muse import MuseMonitor


SERVER = "192.168.43.126"
PORT = 5000

if __name__ == "__main__":
    muse = MuseMonitor(SERVER, PORT)
    muse.run()
