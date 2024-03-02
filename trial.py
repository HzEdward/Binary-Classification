#write a __getiem__ method for the following class so that it can be used as a dictionary
class MyDict:
    def __init__(self):
        self.data = {}
    def __getitem__(self, key):
        return self.data[key]
    def __setitem__(self, key, value):
        self.data[key] = value
    def __delitem__(self, key):
        del self.data[key]
    def keys(self):
        return self.data.keys()
    def values(self):
        return self.data.values()
    def items(self):
        return self.data.items()
    def __str__(self):
        return str(self.data)
    def __len__(self):
        return len(self.data)
