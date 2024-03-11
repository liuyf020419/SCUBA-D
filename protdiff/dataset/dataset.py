import torch
import torch.nn.functional as F
import torch.utils.data as torch_data
import numpy as np
import contextlib
import itertools
import math
import operator
import os
import collections
import time
import queue
from threading import Thread
import logging
logger = logging.getLogger(__name__)

# from alphafold.data.utils.data_transforms import pad_dim

class BaseDataset(torch_data.Dataset):
    def __getitem__(self, index: int):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError

    def reset_data(self):
        """
         reload datalist for change datalist online
        """
        raise NotImplementedError


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count.

    Args:
        iterable (iterable): iterable to wrap
        start (int): starting iteration count. Note that this doesn't
            actually advance the iterator.
        total (int): override the iterator length returned by
            ``__len__``. This can be used to truncate *iterator*.

    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, start=None, total=None):
        self.iterable = iterable
        self.itr = iter(self)

        if start is None:
            self.n = getattr(iterable, "n", 0)
        else:
            self.n = start

        if total is None:
            self.total = self.n + len(iterable)
        else:
            self.total = total

    def __len__(self):
        return self.total

    def __iter__(self):
        for x in self.iterable:
            if self.n >= self.total:
                raise RuntimeError(
                    "Mismatch between actual and expected iterable length. "
                    "This may be caused by resuming training from a checkpoint using "
                    "a different number of GPUs, in which case you can try the "
                    "--reset-dataloader option. Alternatively you may have a train or "
                    "validation set that is smaller than the number of GPUs. If none "
                    "of these apply, please report this to the fairseq developers."
                )
            self.n += 1
            yield x

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        """Whether the iterator has been exhausted."""
        return self.n < len(self)

    def skip(self, num_to_skip):
        """Fast-forward the iterator by skipping *num_to_skip* elements."""
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        return self

    def take(self, n):
        """
        Truncates the iterator to n elements at most.
        """
        self.total = min(self.total, n)

        # Propagate this change to the underlying iterator
        # Only take after what we have already consumed (i.e. after restarting
        # from checkpoint mid epoch, we have to subtract self.n which is the
        # starting point)
        #
        # This to maintain the invariant self.total = self.n + len(iterable),
        # before calling __next__ or __iter__
        propagated_take = max(n - self.n, 0)
        if hasattr(self.iterable, "take"):
            self.iterable.take(propagated_take)
        else:
            self.iterable = itertools.islice(self.iterable, propagated_take)


@contextlib.contextmanager
def numpy_seed(seed,* addl_seeds):
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed= int(hash((seed,*addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])

class set_torch_seed(object):
    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = get_rng_state()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        set_rng_state(self.rng_state)

class GroupedIterator(CountingIterator):
    """Wrapper around an iterable that returns groups (chunks) of items.

    Args:
        iterable (iterable): iterable to wrap
        chunk_size (int): size of each chunk

    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, chunk_size):
        itr = _chunk_iterator(iterable, chunk_size)
        super().__init__(
            itr,
            start=int(math.ceil(getattr(iterable, "n", 0) / float(chunk_size))),
            total=int(math.ceil(len(iterable) / float(chunk_size))),
        )
        self.chunk_size = chunk_size


def _chunk_iterator(itr, chunk_size):
    chunk = []
    for x in itr:
        chunk.append(x)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk



class ShardedIterator(CountingIterator):
    """A sharded wrapper around an iterable, padded to length.

    Args:
        iterable (iterable): iterable to wrap
        num_shards (int): number of shards to split the iterable into
        shard_id (int): which shard to iterator over
        fill_value (Any, optional): padding value when the iterable doesn't
            evenly divide *num_shards* (default: None).

    Attributes:
        n (int): number of elements consumed from this iterator
    """

    def __init__(self, iterable, num_shards, shard_id, fill_value=None):
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError("shard_id must be between 0 and num_shards")
        sharded_len = int(math.ceil(len(iterable) / float(num_shards)))
        itr = map(
            operator.itemgetter(1),
            itertools.zip_longest(
                range(sharded_len),
                itertools.islice(iterable, shard_id, len(iterable), num_shards),
                fillvalue=fill_value,
            ),
        )
        super().__init__(
            itr,
            start=int(math.ceil(getattr(iterable, "n", 0) / float(num_shards))),
            total=sharded_len,
        )


def collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out= None
        # if torch_data.get_worker_info() is not None:
        #     numel = sum([x.numel() for x in batch])
        #     storage = elem.storage()._new_shared(numel)
        #     out = elem.new(storage)
        # return torch.stack(batch, 0, out=out)

        # return torch.stack(pad_batch(batch), 0, out=out)
        try:
            return torch.cat(batch, dim=0, out=out)
        except:
            import pdb; pdb.set_trace()
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in elem}
    raise TypeError('bad collate type {}'.format(type(elem)))


# Object used by _background_consumer to signal the source is exhausted
# to the main thread.
_sentinel = object()


class BackgroundConsumer(Thread):
    def __init__(self, queue, source, max_len):
        Thread.__init__(self)

        self._queue = queue
        self._source = source
        self._max_len = max_len
        self.count = 0

    def run(self):
        try:
            for item in self._source:
                self._queue.put(item)

                # Stop if we reached the maximum length
                self.count += 1
                if self._max_len is not None and self.count >= self._max_len:
                    break

            # Signal the consumer we are done.
            self._queue.put(_sentinel)
        except Exception as e:
            self._queue.put(e)


class BufferedIterator(object):
    def __init__(self, size, iterable):
        self._queue = queue.Queue(size)
        self._iterable = iterable
        self._consumer = None

        self.start_time = time.time()
        self.warning_time = None

        self.total = len(iterable)

    def _create_consumer(self):
        self._consumer = BackgroundConsumer(
            self._queue,
            self._iterable,
            self.total,
        )
        self._consumer.daemon = True
        self._consumer.start()

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def take(self, n):
        self.total = min(self.total, n)

        # Propagate this change to the underlying iterator
        if hasattr(self._iterable, "take"):
            self._iterable.take(n)

    def __next__(self):
        # Create consumer if not created yet
        if self._consumer is None:
            self._create_consumer()

        # Notify the user if there is a data loading bottleneck
        if self._queue.qsize() < min(2, max(1, self._queue.maxsize // 2)):
            if time.time() - self.start_time > 5 * 60:
                if (
                    self.warning_time is None
                    or time.time() - self.warning_time > 15 * 60
                ):
                    logger.debug(
                        "Data loading buffer is empty or nearly empty. This may "
                        "indicate a data loading bottleneck, and increasing the "
                        "number of workers (--num-workers) may help."
                    )
                    self.warning_time = time.time()

        # Get next example
        item = self._queue.get(True)
        if isinstance(item, Exception):
            raise item
        if item is _sentinel:
            raise StopIteration()
        return item

class DataIterator(object):
    def __init__(
        self,
        dataset:BaseDataset,
        num_shards=1,
        shard_id=0,
        epoch=1,
        batch_size=1,
        # frame_size=1000,
        shuffle=True,
        # dynamic_batch=False,
        num_workers=1,
        buffer_size=8,
        # shuffle_size=10000,
        seed= 0,
    ):
        logger.info(f"build data set for rank {shard_id} in {num_shards}, data length is {len(dataset)}")
        self.dataset=dataset
        self.num_shards=num_shards
        self.shard_id= shard_id
        self.batch_size=batch_size
        # self.frame_size=frame_size
        self.shuffle=shuffle
        # self.dynamic_batch=dynamic_batch
        self.num_workers= num_workers
        self.seed= seed
        self.epoch=max(epoch,1)
        # self.shuffle_size=shuffle_size
        self._current_epoch= None
        self._next_epoch=None
        self._batch_sampler= None
        self._dummy_batch= None
        
        # buffer for prefetch
        self.buffer_size= buffer_size

    
    # @staticmethod
    # def collate_fn( batch):
    #     batch = [b for b in batch if b is not None]
    #     if len(batch) == 0:
    #         return {}
    #     cat_data = {}
    #     for name in batch[0].keys():
    #         cat_data[name] = torch.stack([b[name] for b in batch], dim=0)

    #     return cat_data

    @staticmethod  
    def collate_fn(batch):

        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return {}
        cat_data = {}
        max_len = max([b['aatype'].shape[0] for b in batch])

        def pad_dim(data, dim, max_len):
            """ dim int or [int, int]
            """
            if (isinstance(dim, int) or (isinstance(dim, list) and len(dim) == 0)):
                dim = dim
                if isinstance(dim, int):
                    dims = [dim]
                else:
                    dims = dim
            else:
                dims = dim
                dim = dim[0]
                
            def convert_pad_shape(pad_shape):
                l = pad_shape[::-1]
                pad_shape = [item for sublist in l for item in sublist]
                return pad_shape

            shape = [d for d in data.shape]
            assert(shape[dim] <= max_len)

            if shape[dim] == max_len:
                return data

            pad_len = max_len - shape[dim]

            pad_shape = []
            for d in dims:
                tmp_pad_shape = [[0, 0]] * d + [[0, pad_len]] + [[0, 0]] * (len(shape) - d -1)
                pad_shape.append(convert_pad_shape(tmp_pad_shape))

            data_pad = F.pad(data, np.sum(pad_shape, 0).tolist(), mode='constant', value=0)
            return data_pad


        for name in batch[0].keys():
            if name in ['loss_mask', 'len', 'cath_architecture']:
                cat_data[name] = torch.cat([b[name] for b in batch], dim=0)
            elif name in ['pair_res_rel', 'pair_chain_rel', 'ss_adj_pair', 'masked_pair_map']:
                data = torch.cat([pad_dim(b[name], [0, 1], max_len)[None] for b in batch], dim=0)
                cat_data[name] = data
            elif name in ['pdbname', 'noising_mode_idx']:
                data = [b[name] for b in batch]
                cat_data[name] = data
            else:
                data = torch.cat([pad_dim(b[name], 0, max_len)[None] for b in batch], dim=0)
                cat_data[name] = data

        return cat_data

    
    @property
    def batch_sampler(self):
        if self._batch_sampler is None:
            if self.shuffle:
                with numpy_seed(self.seed, self.epoch):
                    # def chunk_shuffle_indices(data_len):
                    #     indices = []
                    #     start = 0
                    #     while start < data_len:
                    #         end = min(data_len, start + self.shuffle_size)
                    #         tmp_indices = np.random.permutation(end - start) + start
                    #         indices.append(tmp_indices)
                    #         start = end
                    #     return np.concatenate(indices, axis=0)
                    # indices = chunk_shuffle_indices(len(self.dataset))
                    indices= np.random.permutation(len(self.dataset))
            else:
                indices= np.arange(len(self.dataset))
            # sampler = [[]]
            # data_sizes = self.dataset.data_sizes() #.seq_lens()
            # pre_batch_size = 0
            # for i in range(len(data_sizes)):
            #     idx = indices[i]
            #     size_ = data_sizes[idx]
            #     if size_ > self.batch_size:
            #         continue

            #     if pre_batch_size + size_ <= self.batch_size:
            #         sampler[-1].append(idx)
            #         pre_batch_size += size_
            #     else:
            #         sampler.append([idx])
            #         pre_batch_size = size_

            # with numpy_seed(self.seed, self.epoch):
            #     batch_index = np.random.permutation(len(sampler))
            #     self._batch_sampler = [sampler[batch_index[i]] for i in range(len(sampler)) ]
            
            self._batch_sampler =list( torch_data.BatchSampler(indices, self.batch_size, drop_last=False))
        return self._batch_sampler
    
    @property
    def dummy_batch(self):
        if self._dummy_batch is None:
            for i in range(len(self.dataset)):
                sample = self.dataset[i]
                
                if sample is not None:
                    self._dummy_batch = self.collate_fn([sample])
                    break
                else:
                    print(f'empyt sample index{i}')
        return self._dummy_batch
    
    def __len__(self):
        return int(math.ceil(len(self.batch_sampler)/float(self.num_shards)))
    
    def end_of_epoch(self):
        return not self._current_epoch.has_next()
    
    @property
    def next_epoch_idx(self):
        if self._next_epoch is not None:
            return self.epoch
        elif self._current_epoch is not None and self.end_of_epoch():
            return self.epoch +1
        else:
            return self.epoch
    
    def next_epoch_itr(self):
        self.epoch= self.next_epoch_idx
        if self._next_epoch is not None:
            logger.info('start from loaded itr')
            self._current_epoch = self._next_epoch
            self._next_epoch = None
        else:
            self._batch_sampler= None
            self._current_epoch = self._get_next_epoch(self.epoch)
        return self._current_epoch

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        if self._current_epoch is not None:
            return self._current_epoch.n
        elif self._next_epoch is not None:
            return self._next_epoch.n
        return 0
    
    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        if self.end_of_epoch():
            epoch = self.epoch + 1
            iter_in_epoch = 0
        else:
            epoch = self.epoch
            iter_in_epoch = self.iterations_in_epoch
        return {
            "epoch": epoch,
            "iterations_in_epoch": iter_in_epoch,
            "shuffle": self.shuffle,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict["epoch"]
        itr_pos = state_dict.get("iterations_in_epoch", 0)
        if itr_pos > 0:
            logger.info(f'continue training ,start from {itr_pos}')
            self._next_epoch = self._get_next_epoch(
                self.epoch,
                offset=itr_pos,
            )
        else:
            self._next_epoch = None
        
    def _get_next_epoch(self, epoch, offset= 0):
        self.dataset.reset_data(epoch)
        batches= self.batch_sampler
        batches= list(
            ShardedIterator(batches, self.num_shards, self.shard_id,fill_value=[])
        )
        logger.info(f'sharded data, batchs for current rank {self.shard_id}|{self.num_shards} is {len(batches)}')
        if offset > 0 and offset >= len(batches):
            logger.info(f'continue training start offset {offset}')
            return None

        if self.num_workers > 0:
            os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
        itr = torch_data.DataLoader(
            self.dataset,
            collate_fn= self.collate_fn,
            batch_sampler=batches[offset:],
            num_workers= self.num_workers
        )
        itr = BufferedIterator(self.buffer_size, itr)
        logger.info(f'start from {offset}, total is {len(itr)}')
        itr = CountingIterator(itr, start= offset)
        return itr



    




