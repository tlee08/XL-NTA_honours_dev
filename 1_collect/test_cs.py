from CaptureServer import *

input(">> Press enter to SETUP...")
setupCapture()

input(">> Press enter to START...")
procs = startCapture("ABCD", "1")

input(">> Press enter to STOP...")
stopCapture(procs)

input(">> Press enter to RESET...")
resetCapture()
