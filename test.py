class MyList:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]


# 创建一个 MyList 对象
my_list = MyList(['a', 'b', 'c'])

# 使用 enumerate 遍历 MyList 对象
for i, item in enumerate(my_list):
    print(i, item)
