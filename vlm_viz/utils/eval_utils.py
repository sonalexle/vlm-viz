import spacy
from bert_score import score as bert_score
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

nlp = None


def tokenize(sent_dict):
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    for key in sent_dict:
        new_sentence_list = []
        for sentence in sent_dict[key]:
            a = ''
            for token in nlp(sentence):
                a += token.text
                a += ' '
            new_sentence_list.append(a.rstrip())
        sent_dict[key] = new_sentence_list

    return sent_dict


def evaluator(gts, res, skip_spice=False):
    """gts: list of list of sentences: ground-truth references
    res: list of list of sentences: predictions (note only 1 sentence in each sublist)
    copied from https://github.com/wangpf3/imagine-and-verbalize/blob/main/verbalization_learning/lib/utils/text_evaluation.py
    """
    eval_dict = {}
    # Todo: use Spacy for tokenization
    gts = tokenize(gts)
    res = tokenize(res)

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    if not skip_spice:
        scorers.append((Spice(), "SPICE"))
    
    # =================================================
    # Compute scores
    # =================================================
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                if m != "Bleu_4": continue
                eval_dict[m] = sc
        else:
            eval_dict[method] = score
    return eval_dict


def compute_bertscore(cand_list, refer_list):
    P_mul, R_mul, F_mul = bert_score(cand_list, refer_list, lang="en", rescale_with_baseline=True)
    return F_mul.mean().item()