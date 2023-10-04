curl "10.0.1.18:8080/serveFile?root=1_collect&fp=GenerateClient.py" > GenerateClient.py
curl "10.0.1.18:8080/serveFile?root=1_collect&fp=CaptureServer.py" > CaptureServer.py
curl "10.0.1.18:8080/serveFile?root=1_collect&fp=HelperFuncs.py" > HelperFuncs.py

curl "10.0.1.18:8080/serveFile?root=1_collect&fp=test_cs.py" > test_cs.py
curl "10.0.1.18:8080/serveFile?root=1_collect&fp=test_gc.py" > test_gc.py
curl "10.0.1.18:8080/serveFile?root=1_collect&fp=test_sockets_c.py" > test_sockets_c.py
curl "10.0.1.18:8080/serveFile?root=1_collect&fp=test_sockets_s.py" > test_sockets_s.py

mkdir -p chrome_extensions
curl "10.0.1.18:8080/serveFile?root=1_collect/chrome_extensions&fp=adblock.crx" > chrome_extensions/adblock.crx
