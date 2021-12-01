`server`
```python
from socket import *

HOST = '127.0.0.1'
PORT = 10000
BUFSIZE = 1024

# Create Server's Socket
ServerSocket = socket(AF_INET, SOCK_STREAM)
ServerSocket.bind((HOST, PORT)); print('Server Socket Binding...')
ServerSocket.listen(100);        print(f'{HOST}:{PORT} Sever Listening Start')

# Accept connection from client
ClientSocket, addr_info = ServerSocket.accept()
print('ACCEPT')
print('Client Information:', ClientSocket)
print('Recieve Client Data : ', ClientSocket.recv(65535).decode())

# Socket close
ClientSocket.close()
ServerSocket.close()
print('Socket Closing...')
```
`client`
```python
from socket import *
import sys

HOST = '127.0.0.1'
PORT = 10000

# Create Client's Socket
ClientSocket = socket(AF_INET, SOCK_STREAM)

try:
    ClientSocket.connect((HOST, PORT))
    ClientSocket.send('I am a client!'.encode())
    print('Connection to the server is successful')

except Exception as e:
    print(f'{HOST}:{PORT}')
    sys.exit()
```


```python
import socket

hostname = socket.gethostname()
socket.gethostbyname(hostname)
```
