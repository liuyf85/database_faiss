from database_faiss import *

# load args
with open('args.json', 'r') as f:
    args_dict = json.load(f)
args = argparse.Namespace(**args_dict)

# creat test data
dstore_size = 100000
dimension = 1024
fp16 = False
if fp16:
    keys = np.random.rand(dstore_size, dimension).astype(np.float16)
    np.save('dstore_fp16_keys.npy', keys)
else:
    keys = np.random.rand(dstore_size, dimension).astype(np.float32)
    np.save('dstore_keys.npy', keys)

DataBase = KNN_Dstore(args)
queries = torch.rand(20, args.dimension)
# get distance and vector of the k closest vectors
dists, knn_vecs = DataBase.get_knns(queries)

print(dists.shape)
print(len(knn_vecs))
print(knn_vecs[0].shape)