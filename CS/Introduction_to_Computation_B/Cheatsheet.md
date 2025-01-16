# Cheatsheet

## 方法

### 乱七八糟

+ 保留几位小数

```python
average_waittime = 431.9
# 格式化为两位小数
formatted_value = f"{average_waittime:.2f}"
print(formatted_value) # 输出431.90
```

+ 把列表打印成一行

```python
output = ' '.join(map(str,index))
print(output)
# 注意⚠️，不能直接把一个数字列表用' '.join(list)写，而需要转换为字符串' '.join(map(str,list))
```

+ next()
  + 主要用于从迭代器中获取下一个元素，如果迭代器已耗尽（没有更多元素可返回），则返回 `default` 值。如果未提供 `default` 值，则在迭代器耗尽时抛出 `StopIteration` 异常。

```python
next(iterator, default)
```

+ 自定义排序 cmp_to_key

```python
from functools import cmp_to_key
def cmp(p1,p2):
  return -1 # p1在p2前
	return 1
	return 0 # 顺序无所谓
list.sort(key = cmp_to_key(cmp))
```

+ 输入的字符的每一位都制成列表

```python
input_string = input("请输入一段文本：")
char_list = list(input_string)
print(char_list)
```

+ 递归可视化，自定义递归调用可视化装饰器

```python
from functools import lru_cache

# 自定义递归调用可视化装饰器
def visualize(func):
    indent = 0

    def wrapper(*args, **kwargs):
        nonlocal indent
        print(f"{'  ' * indent}Entering {func.__name__} with arguments {args}")
        indent += 1
        result = func(*args, **kwargs)
        indent -= 1
        print(f"{'  ' * indent}Exiting {func.__name__} with result {result}")
        return result

    return wrapper

@lru_cache(maxsize=128)
@visualize
def pick(a, b, round):
    if a < b:
        return round
    return pick(a - 1, b, round + 1)

# 测试案例
print(pick(5, 2, 0))
```

+ lru_cache
  + 函数中不能修改global变量
  + lru_cache可能不能全记下来（即便maxsize = None），此时可以使用显式空间缓存，见recursion一节的1117.取石子游戏

```python
from functools import lru_cache

@lru_cache(maxsize=128, typed=False)
def func(...):
    ...
```

+ 传奇函数rotate

```python
def rotate(matrix:list):
    x = len(matrix[0])
    y = len(matrix)
    new_matrix = []
    for j in range(y-1,-1,-1):
        line = []
        for i in range(x):
            line.append(matrix[i][j])
        new_matrix.append(line)
    return new_matrix
```

+ 拷贝

```python
import copy
a = [1,2,3]
b = a.copy() # 浅拷贝
b = copy.copy(a) # 同上
b = list(a) # 同上
c = copy.deepcopy(a) # 深拷贝

# 如果列表中的元素全是不可变对象（如整数、字符串、元组等），浅拷贝=深拷贝。a[1] = 100后，b和c都不变

# 用切片创建浅拷贝
test_region = [row[:] for row in region] # 相当于deepcopy了region
```

+ StringIO

```python
from io import StringIO
data = '''5 8
3 5 1 2 2
4 5 2 1 3'''
d = StringIO(data)
n,m = map(int,d.readline().split())
```

+ 没有结束标志，sys.stdin & try except

```python
import sys
stack = []
min_stack = []
for a in sys.stdin:
    a = a.strip()
    # 后面写程序
```

```PYTHON
while True:
    try:
        a = input()
        # 后面写程序
    except EOFError:
        break
```

+ 加深递归深度

```python
import sys
sys.setrecursionlimit(200000)
```

+ lambda小型函数

```python
# lambda 参数: 表达式
add = lambda x, y: x + y
print(add(3, 5))  # 输出: 8
```

+ groupby

### groupby

+ 注意：只能处理相邻元素！！！
  + 2 6
    4 5
    3 7
    2 6
    2 9
  + 这样的会把几组2分开！！！

```python
from itertools import groupby

data = [(1, 'a'), (1, 'b'), (2, 'c'), (2, 'd'), (3, 'e')]

# 假设 data 已按第一个元素排序
grouped = groupby(data, key=lambda x: x[0])
for key, group in grouped:
    print(key, list(group))
    
# 输出
# 1 [('1', 'a'), ('1', 'b')]
# 2 [('2', 'c'), ('2', 'd')]
# 3 [('3', 'e')]
```

+ **输入：** 一个已排序的可迭代对象和一个可选的键函数。

+ **输出：** 产生一系列的 (key, group) 元组。

  - **`key`：** 组的键值，基于键函数计算。

  - **`group`：** 组内元素的迭代器。

### 字典

基础

```python
person = {}
# 添加元素
person['name'] = 'Alice'
# 修改元素
person['name'] = 'Bob'
# 删除元素
del person['name']

# 使用 pop 删除元素并返回值
age = person.pop("age")

# 返回所有键值对的视图（print字典的方法）
items = person.items()  # dict_items([('name', 'Alice'), ('email', 'alice@example.com')])

# 遍历字典
for key, value in person.items():
    print(f"{key}: {value}")
    
# 清空字典
person.clear()

# 使用 get 方法获取键的值，如果没有则输出"alt"
name = person.get("name","alt")
print(name)  # 输出: Alice
```

```python
# 根据value反查key
my_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 2}

value_to_find = 2
key = next((k for k, v in my_dict.items() if v == value_to_find), None) # 只找第一个
keys = [k for k, v in my_dict.items() if v == value_to_find] # 全部都找

print(keys)  # 输出：['b', 'd']
```

+ **defaultdict**：一种特殊字典，在访问不存在的键时，不会直接报错 `KeyError`，而是根据提供的默认工厂函数自动为该键生成一个默认值。**`defaultdict(int)` 的默认值为 `0`**，适合用于计数器的场景。**在懒更新中使用。**
  + “访问不存在的键”指，直接修改某个键对应的值

```python
from collections import defaultdict

# 创建一个 defaultdict，默认值为 0
count = defaultdict(int)

# 添加计数
count['a'] += 1
count['b'] += 2
count['a'] += 3

print(count)  # 输出：defaultdict(<class 'int'>, {'a': 4, 'b': 2})
```

### 集合

```python
s = {1, 2, 3}
s.add(4)  # 添加元素
print(s)  # 输出: {1, 2, 3, 4}

s.remove(2)  # 移除元素 2
print(s)  # 输出: {1, 3, 4}

s.discard(5)  # 删除不存在的元素，不会报错
print(s)  # 输出: {1, 3, 4}

s.clear()  # 清空集合
print(s)  # 输出: set()
```

### 列表

+ 查找元素在列表的位置 index：**`index()`**：返回元素第一次出现的索引
  + `list.index(element)`

### 全排列、笛卡尔积

```python
from itertools import permutations

data = [1, 2, 3]
result = permutations(data)

print(list(result)) # 输出：[(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]

# 指定排列长度
result = permutations(data, 2) # 输出：[(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
```

```python
from itertools import product, permutations

data = [1, 2]

# 排列，不允许重复
print(list(permutations(data, 2)))
# 输出：[(1, 2), (2, 1)]

# 笛卡尔积，允许重复。repeat是重复的遍数
print(list(product(data, repeat=2)))
# 输出：[(1, 1), (1, 2), (2, 1), (2, 2)]
```

```python
# 02811:熄灯问题
possibilities = list(product([0,1],repeat=6)) # 最后一行所有的开关灯可能
```

### ASCII

```python
#chr(ASCII码)
uppercase = {chr(i): i-64 for i in range(65,91)}
lowercase = {chr(i): i-96 for i in range(97,123)}

#ord(字母)
A_ASCII = ord('A')

#判断字母是大写还是小写：.isupper()
letter = 'A'
is_upper = letter.isupper() #返回True
is_lower = letter.islower() #返回False
```

### bfs用：set, deque和heapq

```python
# deque
from collections import deque
inqueue = set()
inqueue.add((1,1)) # 往集合里加东西用add
queue = deque([]) # 往里加东西就是deque([(1,1,2)])
queue.append()
popped = queue.pop()
queue.extend([1,2,3]) # 往后面加好几个，直接extend

# heapq
import heapq
queue = [] # 往里加东西就是直接加
heapq.heappush(queue,(0,x,y))
heapq.heappop(queue)
heapq.heappushpop(queue,(0,x,y))

# 从可迭代对象中找到 前 n 个最大值 或 前 n 个最小值。
heapq.nlargest(n, iterable, key=None)
heapq.nsmallest(n, iterable, key=None)

 # 合并多个已排序的可迭代对象，返回一个迭代器，产生按排序顺序合并后的元素。此处的iterables可以是列表
heapq.merge(*iterables, key=None, reverse=False)
list1 = [1, 3, 5]
list2 = [2, 4, 6]
list3 = [0, 7, 8]
merged = heapq.merge(list1, list2, list3)
print(list(merged))  # 输出：[0, 1, 2, 3, 4, 5, 6, 7, 8]
```



## 算法

### 懒更新

找到一个不需要被删除的最小值时结束。在最小堆中，除了最小值以外的删不删都无所谓

```python
# 每次要删除x时out[x]+=1
# ls: heap
# out: dict

while ls:
    x = heappop(ls)
    if not out[x]:
        new_min = x
        heappush(ls,x) #不需要弹出的，记得压回去
        break
    out[x]-=1
```

```python
# 22067:快速堆猪
import heapq
from collections import defaultdict
h = []
stack = []
out = defaultdict(int)
while True:
    try:
        a = input()
        if a.startswith('push'):
            _, num = a.split()
            num = int(num)
            heapq.heappush(h,num)
            stack.append(num)
        elif a.startswith('pop'):
            if h:
                out[stack.pop()] += 1
        else:
            while h:
                this = heapq.heappop(h)
                if not out[this]:
                    heapq.heappush(h,this)
                    print(this)
                    break
                out[this] -= 1
    except EOFError:
        break
```

### 桶排序

+ 桶排序（Bucket Sort）是一种基于分布的排序算法，它通过**将输入数据分到有限数量的“桶”（Bucket）中，然后对每个桶中的数据进行单独排序**，最后按桶的顺序将数据合并，从而完成排序。
+ 对于区间问题，桶排序的“桶”通常表示某个位置的右边界或覆盖范围，代码中通过 `ends` 数组实现了这一功能。

```python
# 27104:世界杯只因 greedy 优化（桶排序）
n = int(input())
*lst, = map(int,input().split())
end_for_each_start = [0]*n
for i in range(n):
    left = max(0, i-lst[i])
    right = min(n-1, i+lst[i])
    end_for_each_start[left] = max(end_for_each_start[left], right)

ans = 0

l, r = -1, 0
while r < n-1:
    l, r = r, max(end_for_each_start[l+1:r+1])
    ans += 1

print(ans)
```

### 拓扑排序 Kahn算法 Topological order

+ 对有向无环图（DAG）的顶点进行排序，使：如果在图中有一条从顶点 `u` 指向顶点 `v` 的有向边，则在拓扑排序中，`u` 必须排在 `v` 之前。
+ Kahn算法
  + 计算图中所有顶点的入度。
  + 找到所有入度为 0 的顶点，将它们加入队列。
  + 从队列中取出一个顶点，将它加入拓扑排序结果中，并将它的所有邻接顶点的入度减 1。
    + 如果某个邻接顶点的入度减为 0，将它加入队列。
  + 重复步骤 3，直到队列为空。
  + 如果拓扑排序结果中的顶点个数小于图中顶点的总数，说明图中存在环，无法进行拓扑排序。

```python
from collections import deque

def kahn_topological_sort(n, edges):
    # 构建图的邻接表和入度数组
    graph = [[] for _ in range(n)]
    indegree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1

    # 初始化队列，将所有入度为 0 的顶点加入队列
    queue = deque([i for i in range(n) if indegree[i] == 0])
    topo_order = []

    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # 如果拓扑排序结果的顶点数小于 n，说明存在环
    if len(topo_order) < n:
        return None  # 图中存在环，无法进行拓扑排序

    return topo_order

# 示例
n = 6  # 顶点数
edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]  # 边集合
print(kahn_topological_sort(n, edges))  # 输出：[5, 4, 2, 3, 1, 0]
```

+ DFS算法

```python
def dfs_topological_sort(n, edges):
    # 构建图的邻接表
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    visited = [False] * n
    stack = []

    def dfs(node):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor)
        stack.append(node)

    # 对所有未访问的顶点进行 DFS
    for i in range(n):
        if not visited[i]:
            dfs(i)

    # 栈中的结果需要逆序
    return stack[::-1]
```

```python
# sy417. 有向无环图的最长路的最优方案
# 最坑人的问题是，边的方向并不一定是小边->大边！如果用递归时间复杂度就是O(n+m)，纯dp的话需要先进行拓扑排序（拓扑排序本身的复杂度就已经是O(n+m)）
n,m = map(int,input().split())
graph = [[float('inf')]*n for i in range(n)]
for _ in range(m):
    u,v,w = map(int,input().split())
    graph[u][v] = w

dp = [0]*n
next = [-1]*n
def max_points(i):
    global dp, graph, last
    if dp[i] != 0:
        return dp[i]
    for j in range(n):
        if graph[i][j] != float('inf'):
            newpoints = graph[i][j] + max_points(j)
            if dp[i] < newpoints:
                dp[i] = newpoints
                next[i] = j
    return dp[i]

ans = 0
k = 0
for i in range(n):
    if ans < max_points(i):
        ans = dp[i]
        k = i

ans_seq = []
while k != -1:
    ans_seq.append(k)
    k = next[k]

print('->'.join(map(str,ans_seq)))
```

### 欧拉筛

```python
primes = []
is_prime = [True]*N
is_prime[0] = False;is_prime[1] = False
for i in range(2,N):
    if is_prime[i]:
        primes.append(i)
    for p in primes: #筛掉每个数的素数倍
        if p*i >= N:
            break
        is_prime[p*i] = False
        if i % p == 0: #这样能保证每个数都被它的最小素因数筛掉！
            break
```

### 二分查找 binary search

```python
l = 0;r = N
while l<r:
    mid = (l+r)//2
    if is_valid(mid):
        l = mid+1 # 这样保证闭区间[l,r]内每个数都是未知是否可行的
    else:
        r = mid 
ans = l (=r) # 由于l=r，l和r都是答案
```

+ 典型应用：最优化问题，特别是“最值的最值”问题。这类问题所求的最优值通常具有”单调性质“，即**小于某个数的都可以，但大于它的都不行**。

```python
# 08210:河中跳房子
l,n,m = map(int,input().split())
rock = []
for i in range(n):
    r = int(input())
    rock.append(r)
rock.append(l)

def is_valid(a):
    last = 0
    remove = 0
    for i in range(0,n+1):
        if rock[i] - last < a:
            remove += 1
        else:
            last = rock[i]
        if remove > m:
            return False
    return True

ans = 0
left,right = 0,l+1
while left < right:
    mid = (left+right)//2
    if is_valid(mid):
        ans = mid
        left = mid+1
    else:
        right = mid
print(ans)
```

### Kadane算法

+ 在一个一维整数数组中找到具有最大和的连续子数组。

```python
def max_subarray_sum(arr):
    max_current = max_global = arr[0]
    
    for num in arr[1:]:
        max_current = max(num, max_current + num)
        if max_current > max_global:
            max_global = max_current
            
    return max_global

# 测试用例
arr = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print("最大子数组和为:", max_subarray_sum(arr))  # 输出: 最大子数组和为: 6
```

+ 土豪购物的dp解已经是Kadane算法：分别计算了左侧和右侧具有最大和的连续子数组

### 多重背包的最优化

+ 给定以下输入：`N` 种物品，每种物品有价值 `v[i]`；重量 `w[i]`；数量限制 `c[i]`（最多可选 `c[i]` 个）。背包容量 `W`：背包的最大承载重量。目标：在背包容量允许的条件下，选择若干物品，使得 总价值最大化。

```python
for i in range(1, N + 1):  # 遍历每种物品
    count = c[i]
    k = 1
    while count > 0:
        num = min(k, count)  # 当前组的数量
        weight = num * w[i]
        value = num * v[i]
        for j in range(W, weight - 1, -1):  # 从后往前更新 dp
            dp[j] = max(dp[j], dp[j - weight] + value)
        count -= num
        k *= 2
```

```python
# 20089:NBA门票
n = int(d.readline())
*remain, = map(int,d.readline().split())
price = [1, 2, 5, 10, 20, 50, 100]

if n % 50 != 0:
    print('Fail')
else:
    n = n//50
    dp = [0] + [float('inf')]*n # 花了k块钱时，最少的门票数
    for i in range(7): # 从小票开始买，大票才有一个替换多个的优化空间
        cur = price[i]
        for k in range(n,-1,-1):
            for j in range(1,remain[i]+1):
                if k >= j*cur:
                    dp[k] = min(dp[k], dp[k-j*cur] + j)
                else:
                    break
    ans = dp[n]
```

```python
# 20089:NBA门票 二进制分解
else:
    n = n//50
    dp = [0] + [float('inf')]*n # 花了i块钱时，最少的门票数
    for i in range(7): # 从小票开始买，大票才有一个替换多个的优化空间
        cur = price[i]
        rem = remain[i]
        k = 1
        while k <= rem:
        for k in range(n,-1,-1):
            for j in range(1,remain[i]+1):
                if k >= j*cur:
                    dp[k] = min(dp[k], dp[k-j*cur] + j)
                else:
                    break
    ans = dp[n]
```

### Dijkstra算法

+ 和普通bfs的区别在于，遍历某节点的所有邻接节点时，并不是直接都加入队列，而是先判断从当前节点走是否能让这个临接节点的距离优化。如果能，再加入

```python
function Dijkstra(graph, src):
    dist = [∞, ∞, ..., ∞]  # 初始化距离数组
    dist[src] = 0          # 起点到自身的距离为 0
    visited = []           # 初始化已访问节点集合

    while 未访问节点不为空:
        u = 未访问节点中 dist 最小的节点
        将 u 标记为已访问

        for 邻接节点 v of u:
            if v 未被访问:
                if dist[u] + weight(u, v) < dist[v]:
                    dist[v] = dist[u] + weight(u, v)

    return dist
```

```python
# 20106:走山路（的bfs部分）
		dist = [[float('inf')]*n for _ in range(m)]
    heap = []
    heap.append((0,(startX,startY)))
    dist[startX][startY] = 0
    found = False

    while heap:
        strength,coordinate = heapq.heappop(heap)
        nowX = coordinate[0]
        nowY = coordinate[1]
        if nowX == endX and nowY == endY:
            found = True
            break
        for i in range(4):
            nextX = nowX + dx[i]
            nextY = nowY + dy[i]
            if is_valid(nextX,nextY):
                alt = strength + abs(map_save[nextX][nextY]-map_save[nowX][nowY])
                if alt >= dist[nextX][nextY]:
                    continue
                heapq.heappush(heap,(alt,(nextX,nextY)))
                dist[nextX][nextY] = alt
```



## （以上没有包含的）经典题目

### dp

#### 最长上升子序列

```python
# 02757: 最长上升子序列
n = int(input())
*lne, = map(int,input().split())
dp = [1]*n # 长度小于i的最长上升子序列
for i in range(1,n):
    for j in range(i):
        if lne[i] > lne[j]:
            dp[i] = max(dp[i], dp[j]+1)
print(max(dp))
```

#### 最长公共子序列

```python
# 02806:公共子序列
x,y = input().split()
        xlen = len(x)
        ylen = len(y)
        dp = [[0]*(ylen+1) for _ in range(xlen+1)] # x到第i位，y到第j位时，最长的公共子列长度
        for i in range(1,xlen+1):
            for j in range(1,ylen+1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        print(dp[xlen][ylen])
```

#### 最长上升子序列的最优方案

+ 开另一个数组记每个数的上一个数，最后再爬回去

```python
# sy413. 最长上升子序列的最优方案
n = int(input())
*lst, = map(int,input().split())
dp = [1]*n
last = [-1]*n
ans = 0
ans_i = 0
for i in range(1,n):
    for j in range(i):
        if lst[j] <= lst[i] and dp[i] < dp[j]+1:
            dp[i] = dp[j]+1
            last[i] = j
        if dp[i] > ans:
            ans = max(dp[i],ans)
            ans_i = i
max_seq = []
while ans_i != -1:
    max_seq.append(lst[ans_i])
    ans_i = last[ans_i]
max_seq.reverse()
print(ans)
print(' '.join(map(str,max_seq)))
```

#### 01背包

+ 倒序遍历容量，是为了防止重复选择

```python
# 01背包问题
n,v = map(int,input().split())
*weight, = map(int,input().split())
*value, = map(int,input().split())
mw = max(weight)
dp = [0]*(v+1) # 容量为j时的最大价值
for i in range(n):
    tweight, tvalue = weight[i], value[i]
    for j in range(v,tweight-1,-1):
        dp[j] = max(dp[j], dp[j-tweight] + tvalue)
print(dp[v])
```

+ 二维dp

```python
def knapsack(n, V, weights, values):
    # 初始化dp数组，大小为(n+1) x (V+1)
    dp = [[0] * (V + 1) for _ in range(n + 1)]
    
    # 动态规划求解
    for i in range(1, n + 1):
        for v in range(V + 1):
            if v >= weights[i - 1]:
                dp[i][v] = max(dp[i - 1][v], dp[i - 1][v - weights[i - 1]] + values[i - 1])
            else:
                dp[i][v] = dp[i - 1][v]
    
    # 返回最大价值
    return dp[n][V]
```

#### 01背包问题的最优方案

```python
def knapsack(n, maxW, weights, values):
    # 初始化dp和choice数组
    dp = [[0] * (maxW + 1) for _ in range(n + 2)]
    choice = [[-1] * (maxW + 1) for _ in range(n + 2)]
    
    # 倒序遍历物品
    for i in range(n, 0, -1):
        for v in range(maxW + 1):
            # 不放入物品i
            dp[i][v] = dp[i + 1][v]
            choice[i][v] = 1
            # 如果容量允许，尝试放入物品i
            if v >= weights[i - 1] and dp[i + 1][v - weights[i - 1]] + values[i - 1] >= dp[i + 1][v]:
                dp[i][v] = dp[i + 1][v - weights[i - 1]] + values[i - 1]
                choice[i][v] = 0
    
    # 输出最大价值
    print(dp[1][maxW])
    
    # 回溯选择的物品编号
    chosen_items = []
    v = maxW
    for i in range(1, n + 1):
        if choice[i][v] == 0:
            chosen_items.append(i)
            v -= weights[i - 1]
    
    # 输出选择的物品编号
    print(" ".join(map(str, chosen_items)))
```

#### 完全背包

+ 每个物品可以选任意次

```python
def max_value(n, V, weights, values):
    # 初始化dp数组，dp[v]表示容量为v时的最大价值
    dp = [0] * (V + 1)
    
    # 遍历每个物品
    for i in range(n):
        # 从当前物品的重量开始，直到背包的最大容量
        for v in range(weights[i], V + 1):
            # 状态转移方程
            dp[v] = max(dp[v], dp[v - weights[i]] + values[i])
    
    # 返回背包容量为V时的最大价值
    return dp[V]
```

#### 受到祝福的平方（能不能分解为几个完全平方数拼接）

```python
def is_square(x:str):
    x = int(x)
    if x <= 0:
        return False
    sqrt_x = int(x**0.5)
    return sqrt_x * sqrt_x == x
    
s = input()
l = len(s)
dp = [True] + [False] * l

for i in range(l+1):
    for j in reversed(range(i)):
        if dp[j]:
            curr_n = int(s[j:i])
            if is_square(curr_n):
                dp[i] = True
                break
if dp[l]:
    print("Yes")
else:
    print("No")
```

### recursion

```python
# 1115. 取石子游戏 dfs
def dfs(a,b,memo:dict):
    if b > a:
        a,b = b,a
    if (a,b) in memo:
        return memo[(a,b)]
    if b == 0:
        memo[(a,b)] = False
        return False
    for k in range(1,a//b+1):
        if not dfs(a-b*k,b,memo):
            memo[(a,b)] = True
            return True
    memo[(a,b)] = False
    return False

while True:
    a,b = map(int,input().split())
    if a+b == 0:
        break
    memo = {}
    if dfs(a,b,memo):
        print('win')
    else:
        print('lose')
```

```python
# 02754:八皇后
all_ans = []
ans = [-1]*8
def dfs(ans,line):
    if line == 8:
        all_ans.append(''.join(map(str,[x+1 for x in ans])))
        return
    for col in range(8):
        not_valid = False
        if col in ans:
            continue
        for i in range(line):
            dist = line-i
            if ans[i] == col + dist or ans[i] == col - dist:
                not_valid = True
                break
        if not_valid:
            continue
        ans[line] = col
        dfs(ans,line+1)
        ans[line] = -1
dfs(ans,0)
print(all_ans)
```

```python
# 汉诺塔
def move(start,middle,end,plates):
    global move_times
    if plates == 1:
        print(f"{start}->{end}")
        move_times += 1
    else:
        move(start,end,middle,plates-1)
        print(f"{start}->{end}")
        move_times += 1
        move(middle,start,end,plates-1)

n = int(input())
print(2**n-1)
move('A','B','C',n)
```

### dfs

```python
# 01088:滑雪
r,c = map(int,data.readline().split())
region = []
for i in range(r):
    *line, = map(int,data.readline().split())
    region.append(line)

directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

dp = [[-1 for j in range(c)] for i in range(r)]

def dfs(x,y):
    global dp
    if dp[x][y] != -1:
        return dp[x][y]
    maxl = 1
    for i in range(4):
        nx = x + directions[i][0]
        ny = y + directions[i][1]
        if 0<=nx<r and 0<=ny<c and region[nx][ny] < region[x][y]:
            maxl = max(maxl,1+dfs(nx,ny))
    dp[x][y] = maxl
    return maxl

maxl = 0
for i in range(r):
    for j in range(c):
        maxl = max(maxl,dfs(i,j))

print(maxl)
```

