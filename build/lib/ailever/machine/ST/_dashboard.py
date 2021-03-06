import os
import argparse
from urllib.request import urlretrieve
parser = argparse.ArgumentParser(description="dashboard parser")
parser.add_argument('--HostDash', type=str, required=False, default='PassToken', help="Host : Dashboard")
parser.add_argument('--PortDash', type=str, required=False, default='PassToken', help="Port : Dashboard")
parser.add_argument('--HostDB', type=str, required=False, default='PassToken', help="Host : DataBase")
parser.add_argument('--PortDB', type=str, required=False, default='PassToken', help="Port : DataBase")
parser.add_argument('--HostRV', type=str, required=False, default='PassToken', help="Host : Real-time Visualization")
parser.add_argument('--PortRV', type=str, required=False, default='PassToken', help="Port : Real-time Visualization")
parser.add_argument('--HostR', type=str, required=False, default='PassToken', help="Host : language R")
parser.add_argument('--PortR', type=str, required=False, default='PassToken', help="Port : language R")
args = parser.parse_args()

def dashboard(name='main',
              HostDash=args.HostDash,
              PortDash=args.PortDash,
              HostDB=args.HostDB,
              PortDB=args.PortDB,
              HostRV=args.HostRV,
              PortRV=args.PortRV,
              HostR=args.HostR,
              PortR=args.PortR,
              ):
    if not os.path.isfile(f'{name}.py'):
        urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/machine/ST/'+name+'.py', f'./{name}.py')
        print(f'[AILEVER] The file "{name}.py" has been sucessfully downloaded!')

    os.system(f'python {name}.py \
            --HostDash {HostDash} \
            --PortDash {PortDash} \
            --HostDB {HostDB} \
            --PortDB {PortDB} \
            --HostRV {HostRV} \
            --PortRV {PortRV} \
            --HostR {HostR} \
            --PortR {PortR}')

