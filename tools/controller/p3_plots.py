"""Plot measured latency and geometry with explicit cohort denominators."""
from pathlib import Path


def export_distributions(report: dict, output: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.3, 3.1))
    for name, label, colour in [('fixed', 'Fixed', '#b85c38'), ('controller', 'Adaptive', '#247881')]:
        data = report[name]
        values = sorted(row['median_e2e_ms'] for row in data['ledger'])
        ax.step(values, [(i + 1) / data['pairs'] for i in range(len(values))],
                where='post', label=label, color=colour)
    ax.set_xlabel('Pair median end-to-end latency (ms)')
    ax.set_ylabel('Fraction of all pairs')
    ax.set_title('UDIS-D; failures retained')
    ax.legend()
    fig.tight_layout()
    for suffix in ('pdf', 'svg'):
        fig.savefig(output / ('controller-latency.' + suffix), metadata={'Author': 'Anonymous Authors'}
                    if suffix == 'pdf' else None)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.3, 3.1))
    for name, label, colour in [('hpatches_fixed', 'Fixed', '#b85c38'),
                                ('hpatches_adaptive', 'Adaptive', '#247881')]:
        data = report[name]
        values = sorted(row['mce_px'] for row in data['ledger'] if row['mce_px'] is not None)
        ax.step(values, [(i + 1) / data['pairs'] for i in range(len(values))],
                where='post', label=label, color=colour)
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlabel('Mean corner error (original pixels)')
    ax.set_ylabel('Fraction of all HPatches pairs')
    ax.set_title('Retained homographies\nRejected pairs stay in denominator')
    ax.legend()
    fig.tight_layout()
    for suffix in ('pdf', 'svg'):
        fig.savefig(output / ('hpatches-mce.' + suffix), metadata={'Author': 'Anonymous Authors'}
                    if suffix == 'pdf' else None)
    plt.close(fig)
