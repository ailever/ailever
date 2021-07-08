import os
import argparse
from urllib.request import urlretrieve

parser = argparse.ArgumentParser(description="dashboard parser")
parser.add_argument('--HostDash', type=str, required=False, default='PassToken', help="Host : Dashboard")
parser.add_argument('--PortDash', type=str, required=False, default='PassToken', help="Port : Dashboard")
parser.add_argument('--HostDB', type=str, required=False, default='PassToken', help="Host : DataBase")
parser.add_argument('--PortDB', type=str, required=False, default='PassToken', help="Port : DataBase")
parser.add_argument('--HostJupyter', type=str, required=False, default='PassToken', help="Host : Jupyter")
parser.add_argument('--PortJupyter', type=str, required=False, default='PassToken', help="Port : Jupyter")
parser.add_argument('--HostRV', type=str, required=False, default='PassToken', help="Host : Real-time Visualization")
parser.add_argument('--PortRV', type=str, required=False, default='PassToken', help="Port : Real-time Visualization")
parser.add_argument('--HostR', type=str, required=False, default='PassToken', help="Host : language R")
parser.add_argument('--PortR', type=str, required=False, default='PassToken', help="Port : language R")
args = parser.parse_args()

def run(name='main',
        server=False,
        HostDash=args.HostDash,
        PortDash=args.PortDash,
        HostDB=args.HostDB,
        PortDB=args.PortDB,
        HostJupyter=args.HostJupyter,
        PortJupyter=args.PortJupyter,
        HostRV=args.HostRV,
        PortRV=args.PortRV,
        HostR=args.HostR,
        PortR=args.PortR,
        ):

    print(f"""
    [AILEVER] * Dashboard SetupInfo
    - name : {name}
    - HostDash : {HostDash}
    - PortDash : {PortDash}
    - HostDB : {HostDB}
    - PortDB : {PortDB}
    - HostJupyter : {HostJupyter}
    - PortJupyter : {PortJupyter}
    - HostRV : {HostRV}
    - PortRV : {PortRV}
    - HostR : {HostR}
    - PortR : {PortR}

    * Directly set R studio server ip/port
      - # /etc/rstudio/rserver.conf
    ...
    """)

    if name == 'main':
        if not os.path.isfile(f'{name}.py'):
            urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/'+name+'.py', f'./{name}.py')
            print(f'[AILEVER] The file "{name}.py" has been sucessfully downloaded!')
    elif name[:2] == 'WS':
        if not os.path.isfile(f'{name}.py'):
            urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/work-sheet/'+name+'.py', f'./{name}.py')
            print(f'[AILEVER] The file "{name}.py" has been sucessfully downloaded!')
    elif name[:4] == 'PROJ':
        if not os.path.isfile(f'{name}.py'):
            urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/project/'+name+'.py', f'./{name}.py')
            print(f'[AILEVER] The file "{name}.py" has been sucessfully downloaded!')
 
    # Error
    if server:
        os.system(f'jupyter lab --port {PortJupyter} --ip {HostJupyter} &')
        os.system(f'python -m visdom.server -p {PortRV} --hostname {HostRV} &')
        os.system(f'rstudio-server start')

        print('[On] visdom server')
        print('[On] jupyter server')
        print('[On] R studio server')
        #os.system(f'service postgresql start')

    try:
        os.system(f'python {name}.py \
                --HostDash {HostDash} \
                --PortDash {PortDash} \
                --HostDB {HostDB} \
                --PortDB {PortDB} \
                --HostJupyter {HostJupyter} \
                --PortJupyter {PortJupyter} \
                --HostRV {HostRV} \
                --PortRV {PortRV} \
                --HostR {HostR} \
                --PortR {PortR}')
    except KeyboardInterrupt:
        print('- You must kill jupyter and visdom process signals.')
        os.system('rstudio-server stop')
        #os.system(f'service postgresql stop')

