import traceback
try:
    example = "你好吗"
    a = 1/0
    print(example)
except:
    print(f'"{example}" -> {traceback.format_exc()}')
    print('1')

# a = input("清水人")
# print(a)