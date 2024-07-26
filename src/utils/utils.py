import torch


def multi_hot(batch_labels, label_size):
    """
        Function: batch로 들어온 label list를 multi-hot vector로 변환
                    Hieararchy의 경우 root, end 등의 스페셜 토큰 또한 multi-hot에 포함됨

        * Params:
            * batch_labels: label idx list of one batch, List[List[int]], e.g.  [[1,2],[0,1,3,4]]
            * label_size : number of label set size. If hierarchy, it will be contain special token for hierarchy (e.g. root, [END]...)
        * Return:
            * multi-hot value for classification -> List[List[int]], e.g. [[0,1,1,0,0],[1,1,0,1,1]
    """
    batch_size = len(batch_labels)
    max_length = max([len(sample) for sample in batch_labels])
    aligned_batch_labels = []
    for sample_label in batch_labels:
        if sample_label:
            aligned_batch_labels.append(sample_label + (max_length - len(sample_label)) * [sample_label[0]])
        else:
            aligned_batch_labels.append([0]*max_length)
    aligned_batch_labels = torch.Tensor(aligned_batch_labels).long()
    batch_labels_multi_hot = torch.zeros(batch_size, label_size).scatter_(1, aligned_batch_labels, 1)
    return batch_labels_multi_hot.long()
