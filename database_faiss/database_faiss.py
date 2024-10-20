import argparse
import os
import numpy as np
import faiss
import time
import json
import torch

class KNN_Dstore(object):
    def __init__(self, args):
        self.dimension = args.dimension
        self.k = args.k
        self.metric_type = args.dist

        if args.dstore_fp16:
            keys = np.memmap(args.dstore_mmap+'-fp16_keys.npy', dtype=np.float16, mode='r', shape=(args.dstore_size, args.dimension))
        else:
            keys = np.memmap(args.dstore_mmap+'_keys.npy', dtype=np.float32, mode='r', shape=(args.dstore_size, args.dimension))

        self.vectors = keys

        args.faiss_index = args.faiss_index + '_' + args.dist

        if not os.path.exists(args.faiss_index+".trained"):
            # Initialize faiss index
            if args.dist in ['l2']:
                quantizer = faiss.IndexFlatL2(args.dimension)
                index = faiss.IndexIVFPQ(quantizer, args.dimension,
                    args.ncentroids, args.code_size, 8, faiss.METRIC_L2)
            elif args.dist in ['ip', 'cos']:
                quantizer = faiss.IndexFlatIP(args.dimension)
                index = faiss.IndexIVFPQ(quantizer, args.dimension,
                    args.ncentroids, args.code_size, 8, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = args.probe

            print('Training Index')
            np.random.seed(args.seed)
            random_sample = np.random.choice(np.arange(keys.shape[0]), size=[min(1000000, keys.shape[0])], replace=False)
            start = time.time()
            # Faiss does not handle adding keys in fp16 as of writing this.
            to_train = keys[random_sample].astype(np.float32)
            print('Finished loading train vecs')
            if args.dist == 'cos':
                faiss.normalize_L2(to_train)
            index.train(to_train)
            print('Training took {} s'.format(time.time() - start))

            print('Writing index after training')
            start = time.time()
            faiss.write_index(index, args.faiss_index+".trained")
            print('Writing index took {} s'.format(time.time()-start))

        print('Adding Keys')
        index = faiss.read_index(args.faiss_index+".trained")

        if args.use_gpu:
            print('Moving to GPUs')
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            faiss.index_cpu_to_gpu(res, 0, index, co)
            print('Now index is on GPUs')
            
        start = args.starting_point
        start_time = time.time()
        while start < args.dstore_size:
            end = min(args.dstore_size, start+args.num_keys_to_add_at_a_time)
            to_add = keys[start:end].copy().astype(np.float32)
            if args.dist == 'cos':
                faiss.normalize_L2(to_add)
            index.add_with_ids(to_add, np.arange(start, end))
            start += args.num_keys_to_add_at_a_time

            if (start % 1000000) == 0:
                print('Added %d tokens so far (took %f s)' % (start, time.time() - start_time))
                print('Writing Index', start)
                faiss.write_index(index, args.faiss_index)

        print("Adding total %d keys" % start)
        print('Adding took {} s'.format(time.time() - start_time))
        print('Writing Index')
        start = time.time()
        faiss.write_index(index, args.faiss_index)
        print('Writing index took {} s'.format(time.time()-start))
        self.index = index
        
    
    def get_knns(self, queries):
        q_vecs = queries.detach().cpu().float().numpy()
        if self.metric_type == 'cos':
            faiss.normalize_L2(q_vecs)
        dists, knn_ids = self.index.search(q_vecs, self.k)
        knn_vecs = [self.vectors[knn_id] for knn_id in knn_ids]
        return dists, knn_vecs