"""Reference RGB four-cell VTC: sum of four group means, not their average."""
from .text_align import text_align_loss, AERIAL, GROUND


def full_view_loss(aux):
    text, scale = aux['text'], aux['logit_scale']
    aerial = aux['is_aerial']
    groups = {
        'A_before': (aux['feat_before'], aerial, AERIAL),
        'A_after': (aux['feat_after'], aerial, GROUND),
        'G_before': (aux['feat_before'], ~aerial, GROUND),
        'G_after': (aux['feat_after'], ~aerial, GROUND),
    }
    total, stats = 0.0, {}
    for name, (feat, select, target) in groups.items():
        loss, acc, margin = text_align_loss(
            feat[select].float() @ aux['proj'].float(), text, target, scale)
        total = total + loss
        stats['acc_' + name] = acc.item()
        stats['margin_' + name] = margin.item()
    stats['push'] = stats['margin_A_after'] - stats['margin_A_before']
    stats['drift'] = stats['margin_G_after'] - stats['margin_G_before']
    stats['n_aerial'] = int(aerial.sum())
    stats['n_total'] = int(aerial.numel())
    return total, stats
