# 位数组长度
m = 8
bit_array = [0] * m

# 2个简单哈希函数用于模拟
def hash1(word):
    return len(word) % m

def hash2(word):
    return sum(ord(c) for c in word) % m

# 设置操作
def insert(word):
    bit_array[hash1(word)] = 1
    bit_array[hash2(word)] = 1

# 查询操作
def query(word):
    # 一票否决：只要有一个位置是0，就确定没出现过
    if bit_array[hash1(word)] == 0 or bit_array[hash2(word)] == 0:
        return False  # 一定没出现过
    else:
        return True   # 可能出现过（可能是假阳性）

items = ["cat", "dog"]

for word in items:
    insert(word)
    print(f'设置"{word}"后位数组: {bit_array}')

# 查询示例
queries = ["bird", "god"]

for word in queries:
    result = query(word)
    print(f'查询 "{word}" → {"可能出现过" if result else "一定没出现过"}')
