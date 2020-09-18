from collections import OrderedDict
from .security import SecurityError, sudo

account = input('[AILEVER] Enter your ID : ')
if account in sudo.members():
    passwd = input(f'[AILEVER] Enter password : ')
    if sudo.identify(account, passwd):
        supervisor_id = input(f'[AILEVER] Your account was succesfully logged-in. Welcome to Ailever!')
        supervisor_passwd = input(f'Promulgate values for a better tomorrow!')
    else : raise SecurityError('[AILEVER] Your password is incorrect.')
else : raise SecurityError('[AILEVER] You are not a member of Ailever.')

from .docs import *

