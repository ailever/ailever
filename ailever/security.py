class SecurityError(Exception):
    pass

class Supervisor:
    def __init__(self):
        self.__ailever_verification_code = 'ailever'
        verification = input("[AILEVER] enter ailever verification code : ")
        if self.__ailever_verification_code == verification : pass
        else : raise SecurityError('[AILEVER] permission denied.')
        from collections import OrderedDict
        self.__users = OrderedDict()
        self.__users['sudo'] = 'ailever'

    def enroll(self, account, passwd):
        self.__users[account] = passwd

    def members(self):
        return self.__users.keys()
    
    def identify(self, account, passwd):
        if self.__users[account] == passwd:
            return True
        else:
            return False

sudo = Supervisor()
sudo.enroll('dongmyeong', 'dongmyeong')
sudo.enroll('eunseo', 'eunseo')
sudo.enroll('kyuhun', 'kyuhun')
sudo.enroll('hyehyun', 'hyehyun')

