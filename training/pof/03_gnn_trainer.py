"""
This script trains the GNN Model using dataset prepared by 'prep_data_from_images.py'

The model will be saved to workspace/trained/run{num}

This will save the models as best.pt and last.pt
best.pt is the one that has the best metrics for model evaluation
last.pt is the last saved checkpoint of the model
"""
import os
import sys
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Union
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.utils.helpers import get_base_path
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from core.gnn.model import GNN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')

SEED = 12345
torch.manual_seed(SEED)

PROJECT_ROOT = get_base_path()

workspace = PROJECT_ROOT/ 'workspace'
def get_run_dir(base_dir: Union[str,Path] = workspace / 'trained'):
    """Creates a unique, numbered run directory for storing training artifacts."""
    base_dir = Path(base_dir)
    if not base_dir.exists():
        base_dir.mkdir(parents=True)

    existing_runs = [int(d.name.replace('run', '')) for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('run')]
    last_run = max(existing_runs) if existing_runs else 0
    new_run_dir = base_dir / f'run{last_run + 1}'
    new_run_dir.mkdir()
    return new_run_dir

def plot_results(run_dir, history):
    """Plots and saves the training history."""
    epochs = range(1, len(history['loss']) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training and Validation Metrics')

    # Loss
    axs[0, 0].plot(epochs, history['loss'], 'b-', label='Training Loss')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epochs (x20)')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()

    # POF Accuracy
    axs[0, 1].plot(epochs, history['train_pof_acc'], 'g-', label='Train POF Acc')
    axs[0, 1].plot(epochs, history['test_pof_acc'], 'r-', label='Test POF Acc')
    axs[0, 1].set_title('POF Accuracy')
    axs[0, 1].set_xlabel('Epochs (x20)')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].set_ylim(0, 1.05)

    # Has POF Accuracy
    axs[1, 0].plot(epochs, history['train_has_pof_acc'], 'g-', label='Train Has POF Acc')
    axs[1, 0].plot(epochs, history['test_has_pof_acc'], 'r-', label='Test Has POF Acc')
    axs[1, 0].set_title('Has POF Accuracy')
    axs[1, 0].set_xlabel('Epochs (x20)')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()
    axs[1, 0].set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(run_dir / 'results.png')
    plt.close()

class TopologyDataset(Dataset):
    """Custom Dataset for loading pre-processed graph data."""
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        if not os.path.exists(self.processed_dir):
            return []
        return [f for f in os.listdir(self.processed_dir) if f.endswith('.pt')]

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]), weights_only=False)
        return data

def main():
    # Data Loading
    run_dir = get_run_dir()

    # Configure logger
    file_handler = logging.FileHandler(run_dir / 'train.log')
    file_handler.setFormatter(logging.Formatter('[%(levelname)s]: %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"Results for this run will be saved to: {run_dir}")

    dataset = TopologyDataset(root=str(workspace))

    if len(dataset) == 0:
        logger.error(f"No processed data found in '{workspace / 'processed'}'.")
        logger.info("Run 'prep_data_from_images.py' first to generate the data.")
        sys.exit(1)

    no_edge_graphs = sum(1 for data in dataset if len(data.edge_index) > 2 and data.edge_index.size(1) == 0)
    logger.info(f"Loaded {len(dataset)} graphs, {no_edge_graphs} ({100 * no_edge_graphs / len(dataset):.2f}%) have no edges.")

    dataset = dataset.shuffle()
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    logger.info(f"Training with {len(train_dataset)} graphs, testing with {len(test_dataset)} graphs.")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # --- Model Definition & Data Validation ---
    # Use the dynamic feature size from the first data point
    first_data = dataset[0]
    expected_node_features = first_data.num_node_features
    logger.info(f"Model will be configured for {expected_node_features} node features based on the first data sample.")

    # Validate that all graphs in the dataset have the same number of node features
    for i, data in enumerate(dataset):
        if data.num_node_features != expected_node_features:
            logger.error(f"Inconsistent number of node features found in dataset!")
            logger.error(f"Graph {i} ({data.filename}) has {data.num_node_features} features, but model expects {expected_node_features}.")
            logger.error("Please re-run data preparation to ensure all graphs have a consistent feature set.")
            sys.exit(1)

    # Model Definition
    model = GNN(
        in_channels=expected_node_features,
        hidden_channels=128,
        num_edge_features=first_data.num_edge_features
    )
    logger.info(f'Training for POF prediction with indeterminate handling')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    bce_criterion = nn.BCEWithLogitsLoss()

    def train(loader):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            pof_pred, has_pof_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # Losses
            loss_pof = bce_criterion(pof_pred, batch.y)
            loss_has_pof = bce_criterion(has_pof_pred, batch.has_pof)
            loss = loss_pof + loss_has_pof

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def test(loader):
        model.eval()
        total_pof_acc = 0
        total_has_pof_acc = 0
        total_graphs = 0

        with torch.no_grad():
            for batch in loader:
                pof_pred, has_pof_pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

                # Has POF accuracy
                has_pof_pred_binary = (torch.sigmoid(has_pof_pred) > 0.5).float()
                has_pof_correct = (has_pof_pred_binary == batch.has_pof).sum().item()
                total_has_pof_acc += has_pof_correct
                total_graphs += batch.num_graphs

                # POF accuracy (only for graphs with has_pof=1)
                pof_correct = 0
                for i in range(batch.num_graphs):
                    if batch.has_pof[i] == 0:
                        continue  # Skip indeterminate cases for POF accuracy
                    graph_mask = batch.batch == i
                    graph_pof_pred = pof_pred[graph_mask]
                    graph_pof_y = batch.y[graph_mask]
                    pred_idx = torch.argmax(torch.sigmoid(graph_pof_pred)).item()
                    true_idx = torch.argmax(graph_pof_y).item() if graph_pof_y.sum() > 0 else -1
                    pof_correct += (pred_idx == true_idx)

                total_pof_acc += pof_correct

        has_pof_acc = total_has_pof_acc / total_graphs if total_graphs > 0 else 0
        total_pof_targets = sum(batch.has_pof.sum().item() for batch in loader)
        pof_acc = total_pof_acc / total_pof_targets if total_pof_targets > 0 else 0
        return pof_acc, has_pof_acc

    # Training Loop
    logger.info("\n--- Starting Training ---")
    best_metric = 0.0
    history = {
        'loss': [], 'train_pof_acc': [], 'test_pof_acc': [],
        'train_has_pof_acc': [], 'test_has_pof_acc': []
    }

    for epoch in range(1, 301):
        loss = train(train_loader)
        if epoch % 20 == 0:
            train_pof_acc, train_has_pof_acc = test(train_loader)
            test_pof_acc, test_has_pof_acc = test(test_loader)

            # Log metrics
            print(f'\nEpoch: {epoch:03d}, Loss: {loss:.4f}')
            print(f'  Train | POF Acc: {train_pof_acc:.4f}, Has POF Acc: {train_has_pof_acc:.4f}')
            print(f'  Test  | POF Acc: {test_pof_acc:.4f}, Has POF Acc: {test_has_pof_acc:.4f}')

            # Store history
            history['loss'].append(loss)
            history['train_pof_acc'].append(train_pof_acc)
            history['test_pof_acc'].append(test_pof_acc)
            history['train_has_pof_acc'].append(train_has_pof_acc)
            history['test_has_pof_acc'].append(test_has_pof_acc)

            # Save best model
            current_metric = test_pof_acc + test_has_pof_acc
            if current_metric > best_metric:
                best_metric = current_metric
                best_model_path = run_dir / 'best.pt'
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"New best model saved to {best_model_path}")

    logger.info("\n--- Training Finished ---")
    torch.save(model.state_dict(), run_dir / 'last.pt')
    plot_results(run_dir, history)

    final_pof_acc, final_has_pof_acc = test(test_loader)
    logger.info(f'Final POF Accuracy: {final_pof_acc:.4f}')
    logger.info(f'Final Has POF Accuracy: {final_has_pof_acc:.4f}')
    logger.info(f"All results saved in: {run_dir}")

if __name__ == "__main__":
    main()