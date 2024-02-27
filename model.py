def initialize_model():
    # 加载预训练的ResNet-50模型
    resnet50 = models.resnet50(pretrained=True)

    # 冻结模型的所有参数，避免在训练过程中更新它们
    for param in resnet50.parameters():
        param.requires_grad = False

    # 获取全连接层(fc)的输入特征数量
    num_ftrs = resnet50.fc.in_features

    # 替换全连接层以适应新的分类任务
    resnet50.fc = nn.Linear(num_ftrs, 2)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet50.fc.parameters(), lr=0.001)

    return resnet50, criterion, optimizer