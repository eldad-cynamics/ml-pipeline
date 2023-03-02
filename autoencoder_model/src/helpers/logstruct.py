import json
import os


class Logstruct(object):
    def __init__(self, message, level, account_id=None, stacktrace=None, **kwargs):
        self.message = message
        self.level = level
        self.accountId = account_id
        self.app = os.environ["APPNAME"]
        self.kwargs = kwargs
        if stacktrace is not None:
            self.stacktrace = stacktrace

    def toJSON(self):

        return json.dumps(self.__dict__)

    def __str__(self):
        return self.toJSON()
