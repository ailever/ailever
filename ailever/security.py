class SecurityError(Exception):
    pass


class Supervisor:
    def __init__(self):
        self.__supervisor_password = 'ailever'
        password = input("Enter supervisor password : ")
        if self.__supervisor_password == password : pass
        else : raise SecurityError('You are not supervisor.')
        from collections import OrderedDict
        self.__users = OrderedDict()
        self.__users['ailever'] = 'ailever'

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

