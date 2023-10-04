from CaptureServer import *

makeLogger("test_sockets_s")

# server_sock = startSocket("10.0.2.2", 8080)
server_sock = startSocket("10.0.1.18", 8080)
listenConnection(server_sock, 1)
while True:
    client_sock = acceptConnection(server_sock)

    recv_thread = runThread(recvMessages, (client_sock,))
    sendMessages(client_sock)
    client_sock.close()
