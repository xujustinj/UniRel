import os
import json
import numpy as np
from transformers import PreTrainedTokenizerBase
from tqdm import tqdm
import dataprocess.rel2text

# from transformers import BertTokenizerFast
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

# def save_dict(dict, name):
#     if isinstance(dict, str):
#         dict = eval(dict)
#     with open(f'{name}.txt', 'w', encoding='utf-8') as f:
#         f.write(str(dict))  # dict to str

# def remove_stress_mark(text):
#     text = "".join([c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"])
#     return text

# def change_case(str):
#     s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str)
#     s2 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1)
#     return re.sub(r'[^\w\s]','',s2)

# Driver code


class UniRelDataProcessor(object):
    def __init__(
            self,
            root: str,
            tokenizer: PreTrainedTokenizerBase,
            dataset_name: str = 'nyt',
    ):
        self.task_data_dir = os.path.join(root, dataset_name)
        self.train_path = os.path.join(self.task_data_dir, 'train_split.json')
        self.dev_path = os.path.join(self.task_data_dir, 'valid_data.json')
        self.test_path = os.path.join(self.task_data_dir, 'test_data.json')

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer

        self.label_map_cache_path = os.path.join(self.task_data_dir,
                                                 dataset_name + '.dict')

        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}
        self.max_label_len = 0

        self._get_labels()

        self.pred2text: dict[str, str | int] = {}
        if dataset_name == "nyt":
            self.pred2text = dataprocess.rel2text.nyt_rel2text
            # self.pred2text = {key: "[unused"+str(i+1)+"]" for i, key in enumerate(self.label2id.keys())}
        elif dataset_name == "nyt_star":
            self.pred2text = dataprocess.rel2text.nyt_rel2text
            # self.pred2text = {key: "[unused"+str(i+1)+"]" for i, key in enumerate(self.label2id.keys())}
        elif dataset_name == "webnlg":
            # self.pred2text = {key: "[unused"+str(i+1)+"]" for i, key in enumerate(self.label2id.keys())}
            self.pred2text = dataprocess.rel2text.webnlg_rel2text
            cnt = 1
            exist_value = []
            # Some hard to convert relation directly use [unused]
            for k in self.pred2text:
                v = self.pred2text[k]
                if isinstance(v, int):
                    self.pred2text[k] = f"[unused{cnt}]"
                    cnt += 1
                    continue
                ids = self.tokenizer(v)
                if len(ids["input_ids"]) != 3:
                    print(k, "   ", v)
                if v in exist_value:
                    print("exist", k, "  ", v)
                else:
                    exist_value.append(v)
        elif dataset_name == "webnlg_star":
            # only take the keys from webnlg_rel2text that are in label2id
            for pred in self.label2id.keys():
                try:
                    self.pred2text[pred] = dataprocess.rel2text.webnlg_rel2text[pred]
                except KeyError:
                    print(pred)
            cnt = 1
            exist_value = []
            for k in self.pred2text:
                v = self.pred2text[k]
                if isinstance(v, int):
                    self.pred2text[k] = f"[unused{cnt}]"
                    cnt += 1
                    continue
                ids = self.tokenizer(v)
                if len(ids["input_ids"]) != 3:
                    print(k, "   ", v)
                if v in exist_value:
                    print("exist", k, "  ", v)
                else:
                    exist_value.append(v)
            # self.pred2text = {key: "[unused"+str(i+1)+"]" for i, key in enumerate(self.label2id.keys())}
        self.num_rels = len(self.pred2text.keys())
        self.max_label_len = 1
        self.pred2idx: dict[str, int] = {
            pred: i for i, pred in enumerate(self.pred2text.keys())}
        self.pred_str = " ".join(self.pred2text.values())
        self.idx2pred = {value: key for key, value in self.pred2idx.items()}
        self.num_labels = self.num_rels

    def get_train_sample(
            self,
            token_len: int = 100,
            n_samples: int | None = None,
    ):
        return self._pre_process(
            self.train_path,
            token_len=token_len,
            n_samples=n_samples,
        )

    def get_dev_sample(
            self,
            token_len: int = 150,
            n_samples: int | None = None,
    ) -> dict[str, list]:
        return self._pre_process(
            self.dev_path,
            token_len=token_len,
            n_samples=n_samples,
        )

    def get_test_sample(
            self,
            token_len: int = 150,
            n_samples: int | None = None,
    ) -> dict[str, list]:
        return self._pre_process(
            self.test_path,
            token_len=token_len,
            n_samples=n_samples,
        )

    def get_specific_test_sample(
            self,
            data_path: str,
            token_len: int = 150,
            n_samples: int | None = None,
    ) -> dict[str, list]:
        return self._pre_process(
            data_path,
            token_len=token_len,
            n_samples=n_samples,
        )

    def _get_labels(self) -> None:
        label_num_dict = {}
        # if os.path.exists(self.label_map_cache_path):
        #     label_map = load_dict(self.label_map_cache_path)
        # else:
        label_set: set[str] = set()
        for path in [self.train_path, self.dev_path, self.test_path]:
            fp = open(path)
            samples = json.load(fp)
            for data in samples:
                sample = data
                for spo in sample["relation_list"]:
                    pred: str = spo["predicate"]
                    label_set.add(pred)
                    if pred not in label_num_dict:
                        label_num_dict[pred] = 0
                    label_num_dict[pred] += 1
        label_map = {idx: label for idx, label in enumerate(sorted(label_set))}
        # write_dict(self.label_map_cache_path, label_map)
        # fp.close()
        self.id2label = label_map
        self.label2id = {val: key for key, val in self.id2label.items()}

    def _pre_process(
            self,
            path: str,
            token_len: int,
            n_samples: int | None,
    ) -> dict[str, list]:
        """
        N = n_samples or number of samples in the data
        T = token_len
        R = number of relations

        Returns:
            text: list[str] (N) the original texts
            spo_list: list[tuple[str, str, str]] (N) relational triples
            spo_span_list: list[tuple[tuple[int, int], int, tuple[int, int]] (N)
            head_label: list[np.ndarray[T+R+2 x T+R+2]] (N)
            tail_label: list[np.ndarray[T+R+2 x T+R+2]] (N)
            span_label: list[np.ndarray[T+R+2 x T+R+2]] (N)
        """
        outputs = {
            'text': [],
            "spo_list": [],
            "spo_span_list": [],
            "head_label": [],
            "tail_label": [],
            "span_label": []
        }
        token_len_big_than_100 = 0
        token_len_big_than_150 = 0
        max_token_len = 0
        data: list = json.load(open(path))
        if n_samples is not None:
            data = data[:n_samples]
        label_dict: dict[str, int] = {}
        for line in tqdm(data):
            if len(line["relation_list"]) == 0:
                continue
            text: str = line["text"]
            input_ids: list[int] = self.tokenizer.encode(text)

            token_encode_len = len(input_ids)
            if token_encode_len > 100+2:
                token_len_big_than_100 += 1
            if token_encode_len > 150+2:
                token_len_big_than_150 += 1
            max_token_len = max(max_token_len, token_encode_len)
            if token_encode_len > token_len + 2:
                continue

            spo_list: set[tuple[str, str, str]] = set()
            spo_span_list: set[tuple[tuple[int, int],
                                     int, tuple[int, int]]] = set()
            # [CLS] texts [SEP] rels
            interaction_matrix_size = token_len + 2 + self.num_rels
            interaction_matrix_dim = [
                interaction_matrix_size, interaction_matrix_size]
            head_matrix = np.zeros(interaction_matrix_dim)
            tail_matrix = np.zeros(interaction_matrix_dim)
            span_matrix = np.zeros(interaction_matrix_dim)

            for spo in line["relation_list"]:
                pred: str = spo["predicate"]
                if pred not in label_dict:
                    label_dict[pred] = 0
                label_dict[pred] += 1
                sub: str = spo["subject"]
                obj: str = spo["object"]
                spo_list.add((sub, pred, obj))
                sub_span: tuple[int, int] = tuple(spo["subj_tok_span"])
                obj_span: tuple[int, int] = tuple(spo["obj_tok_span"])
                pred_idx = self.pred2idx[pred]
                plus_token_pred_idx = token_len + 2 + pred_idx
                spo_span_list.add((sub_span, pred_idx, obj_span))

                h_s, h_e = sub_span
                t_s, t_e = obj_span
                # Entity-Entity Interaction
                head_matrix[h_s+1][t_s+1] = 1
                head_matrix[t_s+1][h_s+1] = 1
                tail_matrix[h_e][t_e] = 1
                tail_matrix[t_e][h_e] = 1
                span_matrix[h_s+1][h_e] = 1
                span_matrix[h_e][h_s+1] = 1
                span_matrix[t_s+1][t_e] = 1
                span_matrix[t_e][t_s+1] = 1
                # Subject-Relation Interaction
                head_matrix[h_s+1][plus_token_pred_idx] = 1
                tail_matrix[h_e][plus_token_pred_idx] = 1
                span_matrix[h_s+1][plus_token_pred_idx] = 1
                span_matrix[h_e][plus_token_pred_idx] = 1
                span_matrix[t_s+1][plus_token_pred_idx] = 1
                span_matrix[t_e][plus_token_pred_idx] = 1
                # Relation-Object Interaction
                head_matrix[plus_token_pred_idx][t_s+1] = 1
                tail_matrix[plus_token_pred_idx][t_e] = 1
                span_matrix[plus_token_pred_idx][t_s+1] = 1
                span_matrix[plus_token_pred_idx][t_e] = 1
                span_matrix[plus_token_pred_idx][h_s+1] = 1
                span_matrix[plus_token_pred_idx][h_e] = 1

            outputs["text"].append(text)
            outputs["spo_list"].append(list(spo_list))
            outputs["spo_span_list"].append(list(spo_span_list))
            outputs["head_label"].append(head_matrix)
            outputs["tail_label"].append(tail_matrix)
            outputs["span_label"].append(span_matrix)

        print(max_token_len)
        print(f"more than 100: {token_len_big_than_100}")
        print(f"more than 150: {token_len_big_than_150}")
        return outputs
