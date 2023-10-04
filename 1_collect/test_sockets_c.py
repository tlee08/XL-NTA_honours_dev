from GenerateClient import *

makeLogger("test_sockets_c")

# client_sock = startClientSocket("10.0.2.1", 8080, "10.0.2.2", 8080)
client_sock = startClientSocket("10.0.1.18", 8081, "10.0.1.18", 8080)

recv_thread = runThread(recvMessages, (client_sock,))
sendMessages(client_sock)
client_sock.close()
