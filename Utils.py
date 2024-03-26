import torch


# 定义一个保存模型状态字典的函数
def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)


# 定义一个加载模型状态字典的函数
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    best_val_loss = checkpoint['best_val_loss']
    return model, optimizer, epoch, loss, best_val_loss
