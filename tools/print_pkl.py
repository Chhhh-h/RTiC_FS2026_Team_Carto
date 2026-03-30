import mmcv

# 加载 pkl 文件
results = mmcv.load('/cluster/scratch/pangyi/reto/results/segformer_b5_640x640_ade_160k/eval_results.pkl')

# 查看结果
print(results)
print(len(results))