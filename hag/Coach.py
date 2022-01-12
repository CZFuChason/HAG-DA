import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics

import hag

log = hag.utils.get_logger()


class Coach:

    def __init__(self, trainset, devset, testset, model, opt, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args
        # self.label_to_idx = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
        self.label_to_idx = {'ag':0, 'ap':1, 'cc':2, 'ds':3, 'kn':4, 'ooi':5, 'op':6, 'osi':7, 'ot':8, 'pr':9, 'qu':10, 're':11, 'soi':12, 'sp':13, 'ssi':14, 'su':15, 'th':16}
        self.best_dev_f1 = None
        self.best_dev_acc = None
        self.best_dev_wacc = None
        self.best_epoch = None
        self.best_state = None

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_dev_acc = ckpt["best_dev_acc"]
        self.best_dev_wacc = ckpt["best_dev_wacc"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)
        # Early stopping.
        best_dev_f1, best_epoch, best_state = self.best_dev_f1, self.best_epoch, self.best_state
        best_dev_wacc, best_dev_acc = self.best_dev_wacc, self.best_dev_acc
        # Train
        for epoch in range(1, self.args.epochs + 1):
            self.train_epoch(epoch)
            dev_f1, dev_acc, dev_wacc = self.evaluate()
            log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
            log.info("[Dev set] [acc {:.4f}]".format(dev_acc))
            log.info("[Dev set] [weighted acc {:.4f}]".format(dev_wacc))
            # if best_dev_f1 is None or dev_f1 > best_dev_f1:
            # if best_dev_wacc is None or dev_wacc > best_dev_wacc:
            if best_dev_acc is None or dev_acc > best_dev_acc:
                best_dev_f1 = dev_f1
                best_dev_acc = dev_acc
                best_dev_wacc = dev_wacc
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                log.info("Save the best model.")
            test_f1, test_acc, test_wacc = self.evaluate(test=True)
            log.info("[Test set] [f1 {:.4f}]".format(test_f1))
            log.info("[Test set] [acc {:.4f}]".format(test_acc))
            log.info("[Test set] [wacc {:.4f}]".format(test_wacc))

        # The best
        self.model.load_state_dict(best_state)
        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        dev_f1, dev_acc, dev_wacc = self.evaluate()
        log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
        log.info("[Dev set] [acc {:.4f}]".format(dev_acc))
        log.info("[Dev set] [weighted acc {:.4f}]".format(dev_wacc))
        test_f1, test_acc, test_wacc = self.evaluate(test=True)
        log.info("[Test set] [f1 {:.4f}]".format(test_f1))
        log.info("[Test set] [acc {:.4f}]".format(test_acc))
        log.info("[Test set] [weighted acc {:.4f}]".format(test_wacc))
        
        return best_dev_f1, best_dev_acc, best_dev_wacc, best_epoch, best_state

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()
        # for idx in tqdm(np.random.permutation(len(self.trainset)), desc="train epoch {}".format(epoch)):
        # self.trainset.shuffle()
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            data = self.trainset[idx]
            for k, v in data.items():
                data[k] = v.to(self.args.device)
            nll = self.model.get_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.opt.step()

        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss, end_time - start_time))

    def evaluate(self, test=False):
        dataset = self.testset if test else self.devset
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    data[k] = v.to(self.args.device)
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")
            acc = metrics.accuracy_score(golds, preds)
            wacc = metrics.precision_score(golds, preds, average="weighted")

        # return f1
        return f1, acc, wacc

