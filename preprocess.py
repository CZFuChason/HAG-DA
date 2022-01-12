import argparse

from tqdm import tqdm
import pickle

import hag

log = hag.utils.get_logger()


def dialogue_episod(data, win_size=30, stride=10):
    text_win, speaker_win, label_win = [], [], []
    
    for i in range(0,len(data[2]), stride):
        if i+win_size<len(data[2]):
            text_win.append(data[0][i:i+win_size])
            speaker_win.append(data[1][i:i+win_size])
            label_win.append(data[2][i:i+win_size])
        else:
            text_win.append(data[0][-win_size:])
            speaker_win.append(data[1][-win_size:])
            label_win.append(data[2][-win_size:])
    return [text_win, speaker_win, label_win]
        
        
    
def split():
    hag.utils.set_seed(args.seed)
    
    if args.dataset=='jps':
        train_set = dialogue_episod(pickle.load(open('./pickled_embeddings/jps_train_set.pkl', 'rb')),
                                    args.winsize, args.stride)
        test_set = dialogue_episod(pickle.load(open('./pickled_embeddings/jps_test_set.pkl', 'rb')),
                                   args.winsize, args.stride)
        dev_set = dialogue_episod(pickle.load(open('./pickled_embeddings/jps_dev_set.pkl', 'rb')),
                                  args.winsize, args.stride)
        
    elif args.dataset=='swbd':
        train_set = dialogue_episod(pickle.load(open('./pickled_embeddings/swbd_train_set.pkl', 'rb')),
                                    args.winsize, args.stride)
        test_set = dialogue_episod(pickle.load(open('./pickled_embeddings/swbd_test_set.pkl', 'rb')),
                                   args.winsize, args.stride)
        dev_set = dialogue_episod(pickle.load(open('./pickled_embeddings/swbd_dev_set.pkl', 'rb')),
                                  args.winsize, args.stride)
    
    train, dev, test = [], [], []
    
    
    # for [text, speaker, label] in tqdm(train_set, desc="train"):
    for [text, speaker, label] in tqdm(zip(train_set[0], train_set[1], train_set[2]), desc="train"):
        train.append(hag.Sample(speaker, label, text))
        
    for [text, speaker, label] in tqdm(zip(dev_set[0], dev_set[1], dev_set[2]), desc="dev"):
        dev.append(hag.Sample(speaker, label, text))
        
    for [text, speaker, label] in tqdm(zip(test_set[0], test_set[1], test_set[2]), desc="test"):
        test.append(hag.Sample(speaker, label, text))

    return train, dev, test


def main(args):
    train, dev, test = split()
    log.info("number of train samples: {}".format(len(train)))
    log.info("number of dev samples: {}".format(len(dev)))
    log.info("number of test samples: {}".format(len(test)))
    data = {"train": train, "dev": dev, "test": test}
    hag.utils.save_pkl(data, args.data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data save")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["jps", "swbd"],
                        help="Dataset name.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--winsize", type=int, default=30,
                        help="window size for a dialoge episod.")
    parser.add_argument("--stride", type=int, default=10,
                        help="stride for a dialoge episod.")
    args = parser.parse_args()

    main(args)
