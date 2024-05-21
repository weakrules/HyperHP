import numpy as np
import torch
import torch.utils.data
from utils import PAD


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, data, num_dim):
        """
        Data should be a list of event streams; each event stream is a list of dictionaries;
        each dictionary contains: time_since_start, time_since_last_event, type_event
        """
        seq = []
        max_len = 0

        for i in data.values():
            self.type_time = []
            for dim in range(num_dim):
                self.type_time.append(i[dim])
                seq.append(self.type_time)
                max_len = max(max_len, len(i[dim]))

        for i in seq:
            for dim in range(num_dim):
                i[dim] = [PAD] * (max_len - len(i[dim])) + i[dim]

        self.train_seq = np.array(seq)

        self.length = len(data)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """ Each returned element is a list, which represents an event stream """
        return self.train_seq[idx] #, self.train_type[idx]


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        [PAD] * (max_len - len(inst)) + inst
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)


def pad_type(insts):
    """ Pad the instance to the max seq length in batch. """
    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        [PAD] * (max_len - len(inst)) + inst
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)


def collate_fn(insts):
    """ Collate function, as required by PyTorch. """

    train_time, train_type = list(zip(*insts))
    train_time = pad_time(train_time)
    train_type = pad_type(train_type)

    return train_time, train_type


def get_dataloader(data, batch_size, shuffle=True):
    """ Prepare dataloader. """

    ds = EventData(data)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        # collate_fn=collate_fn,
        shuffle=shuffle
    )
    return dl

def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    data = np.load(opt.dataset, allow_pickle=True).item()

    ds = EventData(data, opt.num_dims)
    trainloader = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=opt.batch_size,
        # collate_fn=collate_fn,
        shuffle=True
    )

    return trainloader



