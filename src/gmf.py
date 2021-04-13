import time

import tensorflow as tf
import tensorflow_datasets as tfds


K = 8
MEAN = 0
STD = 0.01




def _get_dataset() -> tfds: #.Dataset?
    print("Load MovieLens dataset")
    ds = tfds.load('movielens/latest-small-ratings', split='train', shuffle_files=True)
    return ds


def _filter_interactions(ix: dict, ix_limit: int) -> dict:
    print(f"Number of users before filter: {len(ix)}")
    fix = ix.copy()
    for k in list(fix.keys()):
        if len(fix[k]) < 20:
            fix.pop(k)
    print(f"Number of users after filter: {len(fix)}")
    for k in fix.keys():
        print(f"User {k}, number of ratings: {len(fix[k])}")
    return fix


def _get_interactions(dataset: object, num_ix: int, ix_limit: int) -> dict:
    def _to_str(b):
        return str(b, encoding="utf8")

    tinyd = dataset.take(num_ix)
    ixs = {}

    # TODO: Should be tinyd.as_numpy_iterator() for convenience
    for a in tinyd:
        uid = _to_str(a['user_id'].numpy())
        mid = _to_str(a['movie_id'].numpy())
        title = a['movie_title']
        print(f"User: {uid} Title: {title} Id: {mid}")
        if uid not in ixs:
            ixs[uid] = []
        ixs[uid].append(mid)



    ixs_filter = _filter_interactions(ixs, ix_limit)
    return ixs_filter


def get_data(num_iteractions=1000, min_interactions=20):
    print("Get data")
    ds = _get_dataset()
    ixs = _get_interactions(ds, num_iteractions, min_interactions)
    return ixs


def massage_data(ixs: dict):
    """
    Map to internal names
    """
    massage_ixs = {}
    item_ids = list(set([x for v in ixs.values() for x in v]))
    item_map = {i:item_ids.index(i) for i in item_ids}
    user_map = {}
    user_counter = 0
    for user in ixs:
        print(f"Handling u:{user}")
        user_map[user] = user_counter
        massage_ixs[user_counter] = list(map(lambda it: item_map[it], ixs[user]))
        user_counter += 1

    if True:
        # TODO: Debug
        print(ixs, "\n")
        print(user_map, "\n")
        print(item_map, "\n")
        print(massage_ixs)

    # Simpler data, M, N
    return massage_ixs, user_counter, len(item_ids)

def get_params(M, N):
    P = tf.random.normal((M, K), mean=MEAN, stddev=STD, dtype=tf.dtypes.float32)
# N
    Q = tf.random.normal((N, K), mean=MEAN, stddev=STD, dtype=tf.dtypes.float32)

    return P, Q  # Also h

def learn_gmf():
    ixs = get_data()

    ixs_m, M, N = massage_data(ixs)
    P, Q = get_params(M, N)

    print(P, Q)


if __name__ == "__main__":
    print("\n\\\\\\ Let's do this, GMF style \\\\\\n")

    start_time = time.perf_counter()

    learn_gmf()


    end_time = time.perf_counter()

    print(f"Total running time (s): {end_time - start_time}")
