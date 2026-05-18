import cuda.tile as ct
import torch

torch.manual_seed(1021)

@ct.kernel
def matmul(A:ct.Array, B:ct.Array, O:ct.Array,
           tileM: ct.Constant, tileN: ct.Constant, tileK: ct.Constant,
           transposeA: ct.Constant, transposeB: ct.Constant):

    """
    ### 标准的矩阵乘法
    O = A @ B
    ### Input Layout:
    A: [M, K]
    B; [K, N]
    O: [M, N]

    ### block mapping as matrix O:
    [(block_x, 1: TileM), (block_y, 1: TileN)]

    ### parameter:
    transposeA: whether to transpose A
    transposeB: whether to transpose B

    假设: 
    A: [4, 4]
    B: [4, 8]
    O: [4, 8]
    问题：为什么要对tileK进行迭代？ 
    因为矩阵乘法是左行乘右列，必须是左完整的行和右完整的列才能进行乘法运算，所以需要对K维度进行迭代，直到把所有的K维度都计算完，把所有的tileK的计算结果累加到accumulator中，最后把accumulator的结果存储到O中。
    不同的kernel只是存放结果，不能进行累加
    """
    block_idx_x, block_idx_y = ct.bid(0), ct.bid(1)
    # print(f"block_idx_x: {block_idx_x}, block_idx_y: {block_idx_y}")
    K = A.shape[0] if transposeA else A.shape[1]
    num_iter_k = ct.cdiv(K, tileK)

    accumulator = ct.full(shape=(tileM, tileN), fill_value=0.0, dtype=ct.float32)
    for iter_k in range(num_iter_k):
        # 加载数据量（访存）： tileM * tileK + tileN * tileK
        tile_A = ct.load(A, (block_idx_x, iter_k), (tileM, tileK), order="F" if transposeA else "C") #  加载的时候同时转置则order为F，否则为C
        tile_B = ct.load(B, (iter_k, block_idx_y), (tileK, tileN), order="F" if transposeB else "C")

        # 计算量： tileM * tileN * tileK * 2 (乘法和加法)
        # 目标：增大 计算量/访存量 的比值，提升算子效率(含义是每次访存都能进行更多的计算)
        # 增大tileM tileN能够提高比值，而增大tileK却不能提高比值。增大tileK确实能提高计算效率，但是是提升的流水线效率, load和mma 的重叠效率。load和mma并非是顺序执行，而是流水线执行

        # RTX5090 访存带宽 1-2TB/s, 计算能力 100-200TFlops, 计算量/访存量的比值需要达到50-100才能充分利用算力
        accumulator = ct.mma(tile_A, tile_B, accumulator)

    ct.store(O, (block_idx_x, block_idx_y), accumulator.astype(O.dtype))

M, N, K = 128, 4096, 4096
# 我们希望尽可能在不影响并行度的情况下，尽可能的提高tileM, tileN, tileK, 并且优先提高tileM，tileN
# 在5090 A卡或H卡上MN基本都是128
# 切分的tile数量 M、N 应该不能小于SM的数量，否则无法充分利用GPU的并行计算能力
tileM, tileN, tileK = 32, 128, 64

A = torch.rand(size=[M, K], dtype=torch.float16, device="cuda")
B = torch.rand(size=[K, N], dtype=torch.float16, device="cuda")
O = torch.zeros(size=[M, N], dtype=torch.float32, device="cuda")

grid = (ct.cdiv(M, tileM), ct.cdiv(N, tileN))

ct.launch(
    torch.cuda.current_stream(),
    grid, matmul,
    (A, B, O, tileM, tileN, tileK, False, False)
)

real = A @ B
print(O - real)
print(real)