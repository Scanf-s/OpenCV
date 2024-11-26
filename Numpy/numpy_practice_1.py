import numpy as np

arr = ['3.14', 2, 5, 7]
np_arr = np.array(arr)

print(arr)
print(type(arr)) # List

print(np_arr) # String이 포함되어 있는 경우 전부 String element로 변환함
print(type(np_arr)) # numpy.ndarray

np_arr_2 = np.array(arr, dtype=np.float32)
print(np_arr_2) # 강제로 float32 데이터타입으로 변환
print(np_arr_2.ndim) # 1차원 배열이므로 1 출력

np_arr_3 = np.array([[1], [2], [3], [4], [5]])
print(np_arr_3) # 2차원임 (2차원 배열에 5개의 행을 지정했으므로)
print(np_arr_3.ndim)

# 배열의 모양 정보를 알고 싶다 -> shape
# shape[0] : 행 수
# shape[1] : 열 수
np_arr_4 = np.array([[1, 2, 3], [4, 5, 6]]) # 2행 3열 이차원 배열
print(np_arr_4.shape) # (2, 3)
print(np_arr_4.shape[0]) # 2
print(np_arr_4.shape[1]) # 3

# shape은 차원이 높아질수록 튜플 자체의 길이가 늘어남 -> len() 사용하면 차원정보를 알 수 있음
print(len(np_arr_4.shape), "차원")

# 배열의 항목의 수를 구하고 싶다면 size를 사용해야 한다
print(np_arr_4.size) # 6개 요소가 있으므로 6이 출력

random_arr = np.random.random((2, 3))
print("Random Arr", random_arr)

# np.linspace
print(np.linspace(0, 10, 5, endpoint=True)) # 0부터 10까지의 구간을 5개로 나누고 싶다면?

# np.arange
print(np.arange(0, 10, 2)) # 0부터 10까지 2씩 증가하는데, 10을 출력하지는 않음

# 전치행렬은 어떻게 구하냐
np_arr_5 = np_arr_4.reshape(3, 2)
print(np_arr_5)
# 또는 T 속성을 사용해서 전치행렬을 구할 수 있다
np_arr_5 = np_arr_4.reshape(2, 3) # 2, 3행렬에 대해 전치 행렬을 구하려면 일단 Reshape 먼저 적용
print(np_arr_5.T)

# 배열 2개를 연결하고 싶을 때, concatenate
x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
result = np.concatenate([x, y])
print(result)

# 2차원 배열을 연결하려는 경우
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
result = np.concatenate([x, y], axis=0) # 세로로 붙일 때 (행방향)
result2 = np.concatenate([x, y], axis=1) # 가로로 붙일 때 (열방향)
print(result)
print(result2)

# 서로 다른 차원의 행렬은 세로로 붙이려는 경우
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([7, 8, 9])
result = np.vstack([x, y])
print(result)

# 서로 다른 차원의 행렬은 가로로 붙이려는 경우
x = np.array([[1],
              [2]])
y = np.array([[7, 8, 9], [5, 6, 7]])
result = np.hstack([x, y])
print(result)

# np.count_nonzero -> 조건을 넣어서 행렬에 있는 원소 개수 필터링
x = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
print(np.count_nonzero(x < 8)) # 8보다 작은 원소의 개수 -> 8
