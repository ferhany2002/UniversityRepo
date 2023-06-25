# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    torch.zeros(1).cuda()
    print(torch.cuda.is_available())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
