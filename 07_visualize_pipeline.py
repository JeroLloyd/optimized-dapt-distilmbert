import os, sys
import argparse
try:
    from graphviz import Digraph
except Exception:
    print("Missing required package 'graphviz' or Graphviz system binaries.")
    print(f"Install python package into the active venv with:\n  {sys.executable} -m pip install graphviz")
    print("On Windows also install Graphviz system package and ensure 'dot' is on PATH (choco or winget):")
    print("  choco install graphviz -y  OR  winget install --id Graphviz.Graphviz -e --source winget")
    sys.exit(1)
from PIL import Image

from nlp_thesis.utils import get_logger, ensure_dir
logger = get_logger(__name__)


OUT_DIR = '.'

# Flowchart
flow_path = os.path.join(OUT_DIR, 'pipeline_flow.png')

# Build graph using graphviz.Digraph object
dot = Digraph('pipeline', format='png')
dot.attr(rankdir='LR')
dot.node('A', 'LazadaQA\n(Unlabeled)')
dot.node('B', 'DAPT\n(MLM)')
dot.node('C', 'FiReCS\n(Labeled)')
dot.node('D', 'Fine-tune\n(Supervised)')
dot.node('E', 'Optimize & Benchmark')

dot.edge('A', 'B')
dot.edge('B', 'D')
dot.edge('C', 'D')
dot.edge('D', 'E')

# Try to render with Graphviz system 'dot'. If 'dot' not found, fallback to Matplotlib + NetworkX renderer.
try:
    dot.render(filename=flow_path, cleanup=True)
    logger.info(f"Saved pipeline flow diagram to {flow_path}")
except Exception as e:
    logger.warning(f"Graphviz render failed: {e}")
    logger.info("Falling back to Matplotlib + NetworkX rendering...")
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        G = nx.DiGraph()
        G.add_node('A', label='LazadaQA\n(Unlabeled)')
        G.add_node('B', label='DAPT\n(MLM)')
        G.add_node('C', label='FiReCS\n(Labeled)')
        G.add_node('D', label='Fine-tune\n(Supervised)')
        G.add_node('E', label='Optimize & Benchmark')

        G.add_edge('A', 'B')
        G.add_edge('B', 'D')
        G.add_edge('C', 'D')
        G.add_edge('D', 'E')

        pos = {'A': (0, 0), 'B': (1, 0), 'C': (1, -1), 'D': (2, 0), 'E': (3, 0)}
        labels = nx.get_node_attributes(G, 'label')

        plt.figure(figsize=(10, 3))
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2600)
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(flow_path)
        plt.close()
        logger.info(f"Saved pipeline flow diagram (matplotlib fallback) to {flow_path}")
    except Exception as e2:
        logger.warning(f"Fallback rendering failed: {e2}")
        logger.warning("No pipeline flow diagram generated.")

# Collect available artifact images
candidates = ['models/dapt-distilmbert/figure3_masking_example.png',
              'models/dapt-distilmbert/dapt_training_loss.png',
              'models/model_A_finetuned/training_curves.png',
              'models/model_B_finetuned/training_curves.png',
              'models/model_C_finetuned/training_curves.png',
              'thesis_benchmarks.png']

existing = [p for p in candidates if os.path.exists(p)]
logger.info('Found images: %s', existing)

# If there is at least one image, create a simple combined contact sheet
if existing:
    imgs = [Image.open(p).convert('RGB') for p in existing]
    widths, heights = zip(*(i.size for i in imgs))

    max_w = max(widths)
    total_h = sum(heights)

    combined = Image.new('RGB', (max_w, total_h), (255,255,255))
    y = 0
    for im in imgs:
        combined.paste(im, (0,y))
        y += im.size[1]

    combined.save('pipeline_summary.png')
    logger.info('Saved pipeline summary to pipeline_summary.png')
else:
    logger.info('No artifact images found to combine. Run DAPT / finetune / benchmark first.')
