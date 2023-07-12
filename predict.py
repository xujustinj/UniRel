import os
import numpy as np
import torch

from transformers import BertTokenizerFast
from dataprocess.data_extractor import get_e2r, get_span_att
import dataprocess.rel2text
from model.model_transformers import UniRelModel, UniRelOutput

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class UniRel:
    def __init__(
            self,
            model_path: str,
            max_length: int = 128,
            dataset_name: str = "nyt",
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.device = device
        self.model: UniRelModel = UniRelModel.from_pretrained(
            model_path).to(self.device)
        added_token = [f"[unused{i}]" for i in range(1, 17)]
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
            "bert-base-cased",
            additional_special_tokens=added_token,
            do_basic_tokenize=False,
        )
        self.max_length = max_length
        self._get_pred_str(dataset_name)

    def _get_pred_str(self, dataset_name):
        self.pred2text = None
        if dataset_name == "nyt":
            self.pred2text = dataprocess.rel2text.nyt_rel2text
        elif dataset_name == "nyt_star":
            self.pred2text = dataprocess.rel2text.nyt_rel2text
        elif dataset_name == "webnlg":
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
            self.pred2text = dataprocess.rel2text.webnlg_rel2text
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
        else:
            print("dataset name error")
            exit(0)
        self.pred_str = ""
        self.max_label_len = 1
        self.pred2idx = {}
        idx = 0
        for k in self.pred2text:
            self.pred2idx[k] = idx
            self.pred_str += self.pred2text[k] + " "
            idx += 1
        self.num_rels = len(self.pred2text.keys())
        self.idx2pred = {value: key for key, value in self.pred2idx.items()}
        self.pred_str = self.pred_str[:-1]
        self.pred_inputs = self.tokenizer.encode_plus(self.pred_str,
                                                      add_special_tokens=False)

    def _data_process(self, text: str | list[str]) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        # text could be a list of sentences or a single sentence
        if isinstance(text, str):
            text = [text]
        inputs = self.tokenizer.batch_encode_plus(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        batched_input_ids = []
        batched_attention_mask = []
        batched_token_type_ids = []
        for b_input_ids, b_attention_mask, b_token_type_ids in zip(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]):
            input_ids = b_input_ids + self.pred_inputs["input_ids"]
            sep_idx = b_input_ids.index(self.tokenizer.sep_token_id)
            input_ids[sep_idx] = self.tokenizer.pad_token_id
            attention_mask = b_attention_mask + [1]*self.num_rels
            attention_mask[sep_idx] = 0
            token_type_ids = b_token_type_ids + [1]*self.num_rels
            batched_input_ids.append(input_ids)
            batched_attention_mask.append(attention_mask)
            batched_token_type_ids.append(token_type_ids)
        return batched_input_ids, batched_attention_mask, batched_token_type_ids

    def _get_e2r(
            self,
            e2r_pred: torch.Tensor,
    ) -> tuple[dict[int, list[int]], set[tuple[int, int]]]:
        """
        Extract entity-relation (subject-relation) and entity-entity interactions from given Attention Matrix.
        Only Extract the upper-right triangle, so should input transpose of the original
        Attention Matrix to extract relation-entity (relation-object) interactions.
        """
        return get_e2r(
            e2r_pred.cpu().numpy(),
            token_len=self.max_length-2,
        )

    def _get_span_att(
            self,
            span_pred: torch.Tensor,
    ) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
        return get_span_att(
            span_pred.cpu().numpy(),
            token_len=self.max_length-2,
        )

    def _extractor(
            self,
            outputs: UniRelOutput,
            input_ids_list: torch.Tensor,
    ) -> list[list[set[tuple[str, str, str]]]]:
        preds_list = []
        for head_pred, tail_pred, span_pred, input_ids in zip(
            outputs.head_preds,
            outputs.tail_preds,
            outputs.span_preds,
            input_ids_list,
        ):
            pred_spo_text: set[tuple[str, str, str]] = set()
            s_h2r, s2s = self._get_e2r(head_pred)
            s_t2r, _ = self._get_e2r(head_pred.T)
            e_h2r, e2e = self._get_e2r(tail_pred)
            e_t2r, _ = self._get_e2r(tail_pred.T)
            start2span, end2span = self._get_span_att(span_pred)
            for l, r in e2e:
                if l not in e_h2r or r not in e_t2r:
                    continue
                if l not in end2span or r not in end2span:
                    continue
                l_spans, r_spans = end2span[l], end2span[r]
                for l_span in l_spans:
                    for r_span in r_spans:
                        l_s, r_s = l_span[0], r_span[0]
                        if (l_s, r_s) not in s2s:
                            continue
                        if l_s not in s_h2r or r_s not in s_t2r:
                            continue
                        common_rels = set(s_h2r[l_s]) & set(
                            s_t2r[r_s]) & set(e_h2r[l]) & set(e_t2r[r])
                        # l_span_new = (l_span[0]+1, l_span[1])
                        # r_span_new = (r_span[0]+1, r_span[1])
                        l_span_new = (l_span[0], l_span[1])
                        r_span_new = (r_span[0], r_span[1])
                        for rel in common_rels:
                            pred_spo_text.add((
                                self.tokenizer.decode(
                                    input_ids[l_span_new[0]:l_span_new[1]+1]),
                                self.idx2pred[rel],
                                self.tokenizer.decode(
                                    input_ids[r_span_new[0]:r_span_new[1]+1])
                            ))
            preds_list.append(list(pred_spo_text))
        return preds_list

    def predict(self, text: str | list[str]) -> list[list[tuple[str, str, str]]]:
        input_ids, attention_mask, token_type_ids = self._data_process(text)
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, device=self.device)
            attention_mask = torch.tensor(attention_mask, device=self.device)
            token_type_ids = torch.tensor(token_type_ids, device=self.device)
        else:
            input_ids = torch.tensor(
                input_ids, device=self.device).unsqueeze(0)
            attention_mask = torch.tensor(
                attention_mask, device=self.device).unsqueeze(0)
            token_type_ids = torch.tensor(
                token_type_ids, device=self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            results = self._extractor(outputs, input_ids)
        return results


if __name__ == "__main__":
    model_path = "./output/nyt/checkpoint-final"
    unirel = UniRel(model_path, dataset_name="nyt")

    print(unirel.predict("In perhaps the most ambitious Mekong cruise attempt, Impulse Tourism, an operator based in Chiang Mai, Thailand, is organizing an expedition starting in November in Jinghong, a small city in the Yunnan province in China."))
    print(unirel.predict("Adisham Hall in Sri Lanka was constructed between 1927 and 1931 at St Benedicts Monastery , Adisham , Haputhale , Sri Lanka in the Tudor and Jacobean style of architecture"))
    print(unirel.predict([
        "Anson was born in 1979 in Hong Kong.",
        "In perhaps the most ambitious Mekong cruise attempt, Impulse Tourism, an operator based in Chiang Mai, Thailand, is organizing an expedition starting in November in Jinghong, a small city in the Yunnan province in China.",
        "Adisham Hall in Sri Lanka was constructed between 1927 and 1931 at St Benedicts Monastery , Adisham , Haputhale , Sri Lanka in the Tudor and Jacobean style of architecture"
    ]))
    print("end")
