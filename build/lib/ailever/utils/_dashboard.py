import os
import argparse
from urllib.request import urlretrieve
parser = argparse.ArgumentParser(description="dashboard parser")
parser.add_argument('--HostDash', type=str, required=False, default='PassToken', help="host : dashboard")
parser.add_argument('--PortDash', type=str, required=False, default='PassToken', help="port : dashboard")
parser.add_argument('--HostDB', type=str, required=False, default='PassToken', help="host : database")
parser.add_argument('--PortDB', type=str, required=False, default='PassToken', help="port : database")
parser.add_argument('--HostRV', type=str, required=False, default='PassToken', help="host : real-time visualization")
parser.add_argument('--PortRV', type=str, required=False, default='PassToken', help="port : real-time visualization")
parser.add_argument('--HostR', type=str, required=False, default='PassToken', help="host : language r")
parser.add_argument('--PortR', type=str, required=False, default='PassToken', help="port : language r")
args = parser.parse_args()

def dashboard(name='main',
              HostDash=args.HostDash,
              PortDash=args.PortDash,
              HostDB=args.HostDB,
              PortDB=args.PortDB,
              HOSTRV=args.HostRV,
              PortRV=args.PortRV,
              HostR=args.HostR,
              PortR=args.PortR,
              ):
    if not os.path.isfile(f'{name}.py'):
        urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/utils/'+name+'.py', f'./{name}.py')
        print(f'[AILEVER] The file "{name}.py" is downloaded!')
    else:
        raise Exception("[AILEVER] Download is failed.")

    os.system(f'python {name}.py \
            --HostDash {HostDash} \
            --PortDash {PortDash} \
            --HostDB {HostDB} \
            --PortDB {PortDB} \
            --HostRV {HostRV} \
            --PortRV {PortRV} \
            --HostR {HostR} \
            --PortR {PortR}')

