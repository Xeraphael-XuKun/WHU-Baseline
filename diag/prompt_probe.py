"""Stage 0.1 -- do the modality words actually reach CLIP's sentence vector?

    python diag/prompt_probe.py --clip /mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt

No GPU: the text tower is 63M parameters and this encodes about twenty
sentences, so it runs on the dev machine in seconds.

What it answers.  Our text supervision works by putting one word into a fixed
template and treating the resulting sentence vectors as fixed points to align
images against.  That only means anything if changing the word actually moves
the sentence vector.  If "a photo of a rgb person" and "a photo of a thermal
person" come out nearly identical, the anchors are nearly the same point, the
loss is asking images to approach three copies of one target, and no amount of
tuning downstream can help.

Why a calibration group is not optional.  Two sentences that share six of
seven words have a high cosine no matter what the seventh word is -- that is a
property of the text tower, not of our words.  So "0.95" on its own is
unreadable.  The controls below bracket it:

    dog vs cat   two words CLIP knows and considers related  -> the HIGH end
    dog vs car   two words CLIP knows and considers apart    -> the LOW end

A modality group sitting at the dog/cat level means the words are being read
as near-synonyms.  Sitting at the dog/car level means they are being read as
genuinely different things, and the anchors are doing their job.

The second table is blunter still: it strips the modality word out of each
sentence and measures how far that moves the vector.  A word that changes
nothing when removed is contributing nothing when present.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbones.clip_text import (CONTEXT_LENGTH, EOT, SOT,  # noqa: E402
                                       CLIPTextEncoder, build_tokenizer)

# Every group is (label, {name: sentence}).  Names are the axis value being
# varied; the rest of each sentence is held fixed inside a group so the cosine
# is attributable to that one word.
GROUPS = [
    ('abbrev  (文档的缩写组)', {
        'RGB': 'a photo of a RGB view person',
        'NIR': 'a photo of a NIR view person',
        'TIR': 'a photo of a TIR view person',
    }),
    ('ours-mod  (whu_mod_lam50 跑过的)', {
        'RGB': 'a photo of a rgb person',
        'NIR': 'a photo of a infrared person',
        'TIR': 'a photo of a thermal person',
    }),
    ('ours-mcam  (刚写的新措辞)', {
        'RGB': 'a photo of a person taken by a color camera',
        'NIR': 'a photo of a person taken by a infrared camera',
        'TIR': 'a photo of a person taken by a thermal camera',
    }),
    ('ours-view  (text_lam50 的两个锚点)', {
        'aerial': 'a photo of a aerial view person',
        'ground': 'a photo of a ground view person',
    }),
    ('T1  (模态命名式)', {
        'RGB': 'a visible-light color photo of a person',
        'NIR': 'a near-infrared night vision photo of a person',
        'TIR': 'a thermal infrared image of a person',
    }),
    ('T2  (成像特性式)', {
        'RGB': 'a color photo of a person, showing clothing color and texture '
               'in daylight',
        'NIR': 'a grayscale night-time photo of a person, preserving shape and '
               'texture but without color',
        'TIR': 'a thermal image of a person, showing body silhouette and heat '
               'distribution, without color or texture',
    }),
]

# The two calibration lines.  Same template, one word swapped, and words whose
# relationship we are not in any doubt about.
CALIBRATION = ('calibration  (标定：CLIP 一定懂的词)', {
    'dog': 'a photo of a dog',
    'cat': 'a photo of a cat',
    'car': 'a photo of a car',
})

# For the second table: which word to remove from each sentence.  A group with
# no entry here is skipped -- the view group's words are not modality words and
# T1/T2 carry the spectrum across several words, so "remove the word" is not
# defined for them.
STRIP = {
    'abbrev  (文档的缩写组)': {'RGB': 'RGB', 'NIR': 'NIR', 'TIR': 'TIR'},
    'ours-mod  (whu_mod_lam50 跑过的)': {'RGB': 'rgb', 'NIR': 'infrared',
                                         'TIR': 'thermal'},
    'ours-mcam  (刚写的新措辞)': {'RGB': 'color', 'NIR': 'infrared',
                                  'TIR': 'thermal'},
    'ours-view  (text_lam50 的两个锚点)': {'aerial': 'aerial', 'ground': 'ground'},
}


def strip_word(sentence, word):
    """Remove one whitespace-delimited token, leaving the rest untouched.

    Deliberately not a substring replace: 'color' appears twice in T2's RGB
    sentence and inside 'colorful' elsewhere, and silently removing the wrong
    occurrence would make the second table say the opposite of the truth.
    """
    parts = sentence.split()
    if word not in parts:
        raise ValueError('{!r} is not a standalone word of {!r}'.format(word, sentence))
    parts.remove(word)
    return ' '.join(parts)


def encode(text, tok, sentences):
    """Sentences -> (L2-normalised [N, 512] vectors, token counts).

    The token count includes SOT and EOT, which is what has to stay under
    CONTEXT_LENGTH.  T2's sentences are long enough that this is a real check
    rather than a formality -- a truncated sentence loses its tail silently and
    the vector still looks perfectly reasonable.
    """
    ids_list, eots, counts = [], [], []
    for s in sentences:
        ids = [SOT] + tok.encode(s) + [EOT]
        if len(ids) > CONTEXT_LENGTH:
            raise ValueError(
                '{!r} is {} tokens, over CLIP\'s {}; it would be truncated'
                .format(s, len(ids), CONTEXT_LENGTH))
        counts.append(len(ids))
        eots.append(len(ids) - 1)
        ids_list.append(ids + [0] * (CONTEXT_LENGTH - len(ids)))

    t = torch.tensor(ids_list, dtype=torch.long)
    with torch.no_grad():
        emb = text.token_embedding(t)
        feat = text(emb, torch.tensor(eots, dtype=torch.long))
    return F.normalize(feat.float(), dim=-1), counts


def pairwise(vecs):
    """[N, D] normalised -> [N, N] cosine."""
    return (vecs @ vecs.t()).clamp(-1.0, 1.0)


def off_diagonal(mat):
    """The distinct pairs of a symmetric matrix, as a flat list."""
    n = mat.shape[0]
    return [float(mat[i, j]) for i in range(n) for j in range(i + 1, n)]


def report_group(text, tok, label, sentences, width=11):
    names = list(sentences)
    vecs, counts = encode(text, tok, [sentences[k] for k in names])
    cos = pairwise(vecs)

    print('\n  [%s]' % label)
    for k, c in zip(names, counts):
        flag = '  <-- 长' if c > 60 else ''
        print('    %-8s %2d tok  %s%s' % (k, c, sentences[k], flag))
    print('    %-8s' % '' + ''.join('%*s' % (width, k) for k in names))
    for i, k in enumerate(names):
        row = '    %-8s' % k
        for j in range(len(names)):
            row += '%*s' % (width, '--' if i == j else '%.4f' % float(cos[i, j]))
        print(row)
    pairs = off_diagonal(cos)
    print('    %-8s mean %.4f   max %.4f   min %.4f'
          % ('', sum(pairs) / len(pairs), max(pairs), min(pairs)))
    return {'label': label, 'pairs': pairs, 'counts': counts,
            'mean': sum(pairs) / len(pairs), 'max': max(pairs), 'min': min(pairs)}


def report_strip(text, tok, label, sentences, strip):
    """How far does removing the word move the sentence vector?"""
    names = [k for k in sentences if k in strip]
    if not names:
        return None
    full = [sentences[k] for k in names]
    bare = [strip_word(sentences[k], strip[k]) for k in names]
    v_full, _ = encode(text, tok, full)
    v_bare, _ = encode(text, tok, bare)
    cos = (v_full * v_bare).sum(dim=-1).clamp(-1.0, 1.0)
    print('\n  [%s]  去掉那个词之后' % label)
    for k, b, c in zip(names, bare, cos.tolist()):
        print('    %-8s cos %.4f   -> %s' % (k, c, b))
    return {'label': label, 'cos': cos.tolist(), 'names': names}


def main():
    p = argparse.ArgumentParser(description='Stage 0.1 text-side separability')
    p.add_argument('--clip', required=True, help='the ORIGINAL CLIP release, ViT-B-16.pt')
    args = p.parse_args()

    from model.make_model import _read_clip_checkpoint
    text = CLIPTextEncoder()
    text.load_clip(_read_clip_checkpoint(args.clip))
    text.eval()
    tok = build_tokenizer()

    print('\n' + '=' * 78)
    print('Stage 0.1  文本端可分性 -- 冻结 CLIP 文本塔，同一模板只换一个词')
    print('=' * 78)

    cal = report_group(text, tok, *CALIBRATION)
    rows = [cal]
    for label, sentences in GROUPS:
        rows.append(report_group(text, tok, label, sentences))

    print('\n' + '-' * 78)
    print('  去掉模态词的影响  (cos 越接近 1 = 那个词贡献越小)')
    print('-' * 78)
    for label, sentences in GROUPS:
        if label in STRIP:
            report_strip(text, tok, label, sentences, STRIP[label])

    # The calibration pair names are fixed by CALIBRATION above, and both
    # brackets are read off the same matrix that produced everything else.
    cal_cos = pairwise(encode(text, tok, list(CALIBRATION[1].values()))[0])
    hi = float(cal_cos[0, 1])        # dog vs cat  -- near-synonym ceiling
    lo = float(cal_cos[0, 2])        # dog vs car  -- unrelated floor

    print('\n' + '=' * 78)
    print('  汇总                       mean      max      min     判读')
    print('  标定：dog<->cat %.4f (近义上界)   dog<->car %.4f (远义下界)' % (hi, lo))
    print('-' * 78)
    for r in rows:
        if r['label'] == CALIBRATION[0]:
            continue
        m = r['mean']
        # Phrased as a position between the two brackets, low end first.  An
        # earlier version said "距上界 N%", which reads as "close to the
        # ceiling" when it meant the opposite.
        if m >= hi:
            verdict = '几乎没起作用 (贴近 dog/cat 那一端)'
        elif m <= lo:
            verdict = '起作用了 (已在 dog/car 之外)'
        else:
            frac = (m - lo) / (hi - lo) if hi > lo else float('nan')
            verdict = ('起作用了，但只到 dog/car 上方 %.0f%% 处 '
                       '(0%%=dog/car, 100%%=dog/cat)' % (frac * 100))
        print('  %-34s %.4f   %.4f   %.4f   %s'
              % (r['label'], m, r['max'], r['min'], verdict))
    print('=' * 78)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
