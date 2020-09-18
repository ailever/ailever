from collections import OrderedDict
from .security import SecurityError, sudo

account = input('[AILEVER] enter your ID : ')
if account in sudo.members(True):
    supervisor_id = input(f'[AILEVER] your account "{account}" was succesfully inspected.')
    passwd = input(f'[AILEVER] enter password : ')
    if sudo.identify(account, passwd):
        supervisor_passwd = input(f'[AILEVER] your account "{account}" was succesfully. Welcome to Ailever!')
        if sudo.identify(supervisor_id, supervisor_passwd):
            pass
    else : raise SecurityError('[AILEVER] your password is incorrect.')
else : raise SecurityError('[AILEVER] you are not a member of Ailever.')

from .docs import *

