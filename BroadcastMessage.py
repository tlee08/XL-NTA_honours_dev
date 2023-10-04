from bottle import Bottle, run, static_file, request
import socket


app = Bottle()


def getIP():
    # Create a temporary socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Connect to a well-known server: Google
    sock.connect(("8.8.8.8", 80))
    # Get the socket's IP address
    ip_addr = sock.getsockname()[0]
    # Close the temporary socket
    sock.close()
    return ip_addr


@app.error(404)
def error404(error):
    return "Oops! Page not found.\n"


@app.route("/")  # type: ignore
def hello():
    return "hello there!\n"


@app.route("/serveIP")  # type: ignore
def serveIP():
    return f"{getIP()}\n"


@app.route("/serveFile")  # type: ignore
def serveFile():
    fp = request.query.get("fp")  # type: ignore
    root = request.query.get("root")  # type: ignore
    if fp and root:
        return static_file(filename=fp, root=root)
    else:
        return (
            "Need to specify the query string parameters, fp and root. \n"
            + "For example, request '10.0.1.18:8080/getFile?fp=filename&root=path/to/folder'\n"
        )


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=8080)
